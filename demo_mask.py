"""
Demo script: bbox crop -> rotate upright -> SMPL mesh render -> mask output + PCK.

For ceiling-mounted fisheye cameras, the person's orientation in the crop depends
on their position relative to the image center. We rotate the crop so the person
appears upright before feeding to the model, then rotate the rendered mesh back.

For each detected person, outputs:
  - {image}_person{i}_vis.jpg   : 3-panel (original | full image + mesh | centered mask)
  - {image}_person{i}_mask.png  : binary mask (white silhouette on black)

Usage:
    python demo_mask.py \
        --image_folder /path/to/images \
        --output_folder /path/to/output \
        --ckpt_path /path/to/checkpoint.ckpt \
        --gt_npz data/test-labels/own_omni_action16_val.npz \
        --gt_root /mnt/dst_datasets/own_omni_dataset/action16_2022/rawframes
"""
import argparse
import cv2
import os
import torch
import smplx
import numpy as np
from glob import glob
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig

from core.utils.utils_detectron2 import DefaultPredictor_Lazy
from core.datasets.dataset import Dataset
from core.utils import recursive_to
from core.utils.renderer_cam import render_overlay_image
from core.utils.train_utils import perspective_projection, trans_points2d_parallel
from core.utils.eval_utils import pck_accuracy
from core.cam_model.fl_net import FLNet
from core.constants import (
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS,
    SMPL_MODEL_PATH, CAM_MODEL_CKPT, DETECTRON_CFG, DETECTRON_CKPT,
)

DEFAULT_CKPT = '/mnt/data_hdd/fzhi/CameraHMR/ckpt/train/runs/mixed_coco/checkpoints/epoch=14-step=210000.ckpt'
DEFAULT_GT_NPZ = 'data/test-labels/own_omni_action16_val.npz'
DEFAULT_GT_ROOT = '/mnt/dst_datasets/own_omni_dataset/action16_2022/rawframes'


def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img
    return aspect_ratio, final_img


def convert_to_full_img_cam(pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    tz = 2.0 * focal_length / (bbox_height * s)
    cx = 2.0 * (bbox_center[:, 0] - (img_w / 2.0)) / (s * bbox_height)
    cy = 2.0 * (bbox_center[:, 1] - (img_h / 2.0)) / (s * bbox_height)
    return torch.stack([tx + cx, ty + cy, tz], dim=-1)


def compute_fisheye_rotation(person_x, person_y, img_center_x, img_center_y):
    """Compute rotation angle to make a person in a ceiling fisheye image appear upright."""
    dx = person_x - img_center_x  # from center to person (head direction)
    dy = person_y - img_center_y
    alpha = np.arctan2(dy, dx)
    theta = alpha + np.pi / 2
    return np.degrees(theta)


def load_gt_data(npz_path, gt_root):
    """Load ground truth from npz. Returns dict: abs_img_path -> list of (center, scale, gtkps)."""
    data = np.load(npz_path, allow_pickle=True)
    imgnames = data['imgname']
    centers = data['center']
    scales = data['scale']
    gtkps = data['gtkps']

    gt_map = {}
    for i in range(len(imgnames)):
        abs_path = os.path.join(gt_root, str(imgnames[i]))
        abs_path = os.path.normpath(abs_path)
        if abs_path not in gt_map:
            gt_map[abs_path] = []
        gt_map[abs_path].append({
            'center': centers[i],
            'scale': scales[i],
            'gtkps': gtkps[i],
        })
    return gt_map


def match_det_to_gt(det_center, gt_entries):
    """Match a detected bbox center to the closest GT entry."""
    if not gt_entries:
        return None
    best_dist = float('inf')
    best_gt = None
    for gt in gt_entries:
        dist = np.linalg.norm(det_center - gt['center'])
        if dist < best_dist:
            best_dist = dist
            best_gt = gt
    return best_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default=DEFAULT_CKPT)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--gt_npz', type=str, default=DEFAULT_GT_NPZ,
                        help='GT npz file for PCK evaluation')
    parser.add_argument('--gt_root', type=str, default=DEFAULT_GT_ROOT,
                        help='Root directory of GT images (rawframes)')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # --- Load GT for PCK ---
    gt_map = {}
    has_gt = False
    if args.gt_npz and os.path.exists(args.gt_npz):
        gt_map = load_gt_data(args.gt_npz, args.gt_root)
        has_gt = True
        print(f'Loaded GT: {sum(len(v) for v in gt_map.values())} annotations for {len(gt_map)} images')

    # --- Init model ---
    from core.camerahmr_model import CameraHMR
    model = CameraHMR.load_from_checkpoint(args.ckpt_path, strict=False, model_type='smpl')
    model = model.to(device)
    model.eval()

    # --- Init detector ---
    detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
    detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = args.threshold
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # --- Init camera model ---
    cam_model = FLNet()
    cam_ckpt = torch.load(CAM_MODEL_CKPT)['state_dict']
    cam_model.load_state_dict(cam_ckpt)
    cam_model.eval()

    # --- Init body model ---
    body_model = smplx.SMPLLayer(model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS).to(device)

    normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    # --- PCK accumulators ---
    all_pck_005 = []
    all_pck_01 = []

    # --- Gather images ---
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    images_list = sorted([p for ext in exts for p in glob(os.path.join(args.image_folder, ext))])
    print(f'Found {len(images_list)} images')

    for img_path in images_list:
        img_cv2 = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        fname = os.path.splitext(os.path.basename(img_path))[0]
        abs_img_path = os.path.normpath(os.path.abspath(img_path))
        full_img_h, full_img_w = img_rgb.shape[:2]
        print(f'Processing {img_path}')

        # Detect on full image
        det_out = detector(img_rgb)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        if len(boxes) == 0:
            print(f'  No person detected, skipping.')
            continue

        bbox_scale_all = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        bbox_center_all = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # Camera intrinsics from full image (just need focal length)
        _, img_full_resized = resize_image(img_rgb, IMAGE_SIZE)
        img_full_t = np.transpose(img_full_resized.astype('float32'), (2, 0, 1)) / 255.0
        img_full_t = normalize_img(torch.from_numpy(img_full_t).float())
        estimated_fov, _ = cam_model(img_full_t.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (full_img_h / (2 * torch.tan(vfov / 2))).item()

        # Look up GT for this image
        gt_entries = gt_map.get(abs_img_path, [])

        # Process each person individually
        n_persons = len(boxes)
        for pi in range(n_persons):
            cx, cy = bbox_center_all[pi]
            bw = boxes[pi, 2] - boxes[pi, 0]
            bh = boxes[pi, 3] - boxes[pi, 1]

            # --- Compute rotation angle for ceiling fisheye ---
            theta_deg = compute_fisheye_rotation(
                cx, cy, full_img_w / 2.0, full_img_h / 2.0)

            # --- Crop around bbox (larger padding to accommodate rotation) ---
            crop_side = max(bw, bh) * 1.8
            half = int(crop_side / 2)
            x1 = max(int(cx - half), 0)
            y1 = max(int(cy - half), 0)
            x2 = min(int(cx + half), full_img_w)
            y2 = min(int(cy + half), full_img_h)
            crop_img = img_rgb[y1:y2, x1:x2]
            crop_h, crop_w = crop_img.shape[:2]
            if crop_h == 0 or crop_w == 0:
                print(f'  Person {pi}: empty crop, skipping.')
                continue

            # --- Rotate crop so person appears upright ---
            rot_center = (crop_w / 2.0, crop_h / 2.0)
            rot_mat = cv2.getRotationMatrix2D(rot_center, theta_deg, 1.0)
            crop_rotated = cv2.warpAffine(crop_img, rot_mat, (crop_w, crop_h),
                                          borderMode=cv2.BORDER_REPLICATE)

            # Transform bbox center through rotation
            new_cx = cx - x1
            new_cy = cy - y1
            rotated_pt = rot_mat @ np.array([new_cx, new_cy, 1.0])
            new_center_rot = np.array([[rotated_pt[0], rotated_pt[1]]])
            new_scale = bbox_scale_all[pi:pi+1]

            # cam_int for rotated crop
            crop_cam_int = np.array([[fl_h, 0, crop_w / 2.0],
                                     [0, fl_h, crop_h / 2.0],
                                     [0, 0, 1]], dtype=np.float32)

            # Dataset on rotated crop
            ds = Dataset(crop_rotated, new_center_rot, new_scale, crop_cam_int, False, img_path)
            dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

            for batch in dl:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out_smpl_params, out_cam, focal_length_ = model(batch)

                # SMPL forward
                smpl_output = body_model(**{k: v.float() for k, v in out_smpl_params.items()})
                pred_vertices = smpl_output.vertices
                pred_keypoints_3d = smpl_output.joints

                b_img_h, b_img_w = batch['img_size'][0]
                cam_trans = convert_to_full_img_cam(
                    pare_cam=out_cam,
                    bbox_height=batch['box_size'],
                    bbox_center=batch['box_center'],
                    img_w=b_img_w,
                    img_h=b_img_h,
                    focal_length=batch['cam_int'][:, 0, 0],
                )

                fl = focal_length_[0].item()
                camera_center = (crop_w / 2.0, crop_h / 2.0)
                cam_t_np = cam_trans[0].cpu().numpy()
                verts_np = pred_vertices[0].cpu().numpy()

                # --- Render on rotated crop (person is upright here) ---
                crop_rot_f = crop_rotated.astype(np.float32) / 255.0
                overlay_rot, valid_mask_rot = render_overlay_image(
                    image=crop_rot_f,
                    camera_translation=cam_t_np.copy(),
                    vertices=verts_np,
                    camera_rotation=None,
                    focal_length=(fl, fl),
                    camera_center=camera_center,
                    mesh_color='pinkish',
                    alpha=1.0,
                    faces=body_model.faces,
                    sideview_angle=0,
                    add_ground_plane=False,
                    correct_ori=True,
                )
                mesh_mask_rot = valid_mask_rot[:, :, 0].astype(np.float32)

                # --- Rotate overlay and mask back to original crop orientation ---
                inv_rot_mat = cv2.getRotationMatrix2D(rot_center, -theta_deg, 1.0)
                overlay_back = cv2.warpAffine(overlay_rot, inv_rot_mat, (crop_w, crop_h))
                mesh_mask_back = cv2.warpAffine(mesh_mask_rot, inv_rot_mat, (crop_w, crop_h))

                # --- Panel 2: paste mesh onto full original image ---
                full_overlay = img_rgb.astype(np.float32) / 255.0
                paste_h = min(y2 - y1, overlay_back.shape[0])
                paste_w = min(x2 - x1, overlay_back.shape[1])
                for c in range(3):
                    region = full_overlay[y1:y1+paste_h, x1:x1+paste_w, c]
                    mesh_region = overlay_back[:paste_h, :paste_w, c]
                    mask_region = mesh_mask_back[:paste_h, :paste_w]
                    full_overlay[y1:y1+paste_h, x1:x1+paste_w, c] = np.where(
                        mask_region > 0.5, mesh_region, region)

                # --- Panel 3: centered mask ---
                ys, xs = np.where(mesh_mask_back > 0.5)
                canvas = np.zeros((crop_h, crop_w), dtype=np.float32)
                if len(ys) > 0:
                    my1, my2 = ys.min(), ys.max() + 1
                    mx1, mx2 = xs.min(), xs.max() + 1
                    mask_patch = mesh_mask_back[my1:my2, mx1:mx2]
                    ph, pw = mask_patch.shape
                    oy = (crop_h - ph) // 2
                    ox = (crop_w - pw) // 2
                    canvas[oy:oy + ph, ox:ox + pw] = mask_patch
                centered_mask_3ch = np.repeat(canvas[:, :, None], 3, axis=2)

                # --- Build 3-panel vis: original | full+mesh | centered mask ---
                vis_h = crop_h
                orig_resized = cv2.resize(
                    img_rgb.astype(np.float32) / 255.0,
                    (int(full_img_w * vis_h / full_img_h), vis_h),
                )
                overlay_resized = cv2.resize(
                    full_overlay,
                    (int(full_img_w * vis_h / full_img_h), vis_h),
                )
                vis = np.concatenate([orig_resized, overlay_resized, centered_mask_3ch], axis=1)
                vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
                binary_mask = (canvas * 255).astype(np.uint8)

                # Save
                vis_path = os.path.join(args.output_folder, f'{fname}_person{pi}_vis.jpg')
                mask_path = os.path.join(args.output_folder, f'{fname}_person{pi}_mask.png')
                cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                cv2.imwrite(mask_path, binary_mask)
                print(f'  Saved: {vis_path}')

                # --- PCK computation ---
                if has_gt and gt_entries:
                    rotation = torch.eye(3, device=device).unsqueeze(0)
                    joints2d = perspective_projection(
                        pred_keypoints_3d,
                        rotation=rotation,
                        translation=cam_trans,
                        cam_intrinsics=batch['cam_int'],
                    )
                    pred_kp = trans_points2d_parallel(joints2d, batch['_trans'])
                    pred_kp = pred_kp / IMAGE_SIZE - 0.5

                    det_c = bbox_center_all[pi]
                    matched_gt = match_det_to_gt(det_c, gt_entries)
                    if matched_gt is not None:
                        raw_kps = matched_gt['gtkps'].copy()  # (44, 3)
                        # GT kps: full_image -> crop -> rotate -> 256x256
                        gt_xy = raw_kps[:, :2].copy()
                        gt_xy[:, 0] -= x1
                        gt_xy[:, 1] -= y1
                        # Apply rotation (same as crop image)
                        ones = np.ones((gt_xy.shape[0], 1))
                        gt_homo = np.hstack([gt_xy, ones])
                        gt_rotated_xy = (rot_mat @ gt_homo.T).T  # (44, 2)
                        # Apply _trans to get to 256x256 space
                        ones2 = np.ones((gt_rotated_xy.shape[0], 1))
                        gt_aug = np.hstack([gt_rotated_xy, ones2])
                        trans_mat = batch['_trans'][0].cpu().numpy()
                        kp_transformed = (trans_mat @ gt_aug.T).T
                        kp_transformed = kp_transformed / IMAGE_SIZE - 0.5

                        gt_44 = np.zeros((1, 44, 3), dtype=np.float32)
                        gt_44[0, :, :2] = kp_transformed
                        gt_44[0, :, 2] = raw_kps[:, 2]
                        gt_kp = torch.tensor(gt_44, device=device, dtype=torch.float32)
                        mask_vis = gt_kp[:, :, 2] > 0

                        if mask_vis[:, :18].any():
                            pck1, _, _ = pck_accuracy(
                                pred_kp[:, :18, :2], gt_kp[:, :18, :2], mask_vis[:, :18], 0.05)
                            pck2, _, _ = pck_accuracy(
                                pred_kp[:, :18, :2], gt_kp[:, :18, :2], mask_vis[:, :18], 0.1)
                            all_pck_005.append(pck1)
                            all_pck_01.append(pck2)

    # --- Final PCK report ---
    print('\n' + '=' * 40)
    if all_pck_005:
        avg_005 = torch.cat(all_pck_005).mean().item()
        avg_01 = torch.cat(all_pck_01).mean().item()
        print(f'  avgpck_0.05: {avg_005:.4f}')
        print(f'  avgpck_0.1:  {avg_01:.4f}')
    else:
        print('  No GT matched — PCK not computed.')
    print('=' * 40)
    print('Done.')


if __name__ == '__main__':
    main()
