#!/usr/bin/env python3
"""
Prepare custom dataset from /mnt/data_hdd/fzhi/output/ to CameraHMR training format.

Reads from TWO sources per sequence:
  1) smpl/<view>/annotations/<subj>_<action>_annotations.json
     → SMPL params (global_orient, body_pose, betas, trans), camera (fx,fy,cx,cy),
       joints_2d (22x2), visibility (22)
  2) joint/npy/transformed/<frame>.npy
     → 3D joints in world/transformed coords (22, 3)

Output: .npz files for CameraHMR with keys:
  imgname, scale, center, pose_cam, shape, cam_int, gtkps, cam_ext, trans_cam, joints_3d_cam

Supports automatic train/val split (--val_ratio or --val_actions).

Examples:
  # All subjects, 90/10 train/val split
  python scripts/prepare_fzhi_custom_data.py \
    --data_root /mnt/data_hdd/fzhi/output \
    --subjects 100832,100837,101010,101020,101032,101412,101883,102023,203915,204124,avatarrex_zzr \
    --val_ratio 0.1

  # Specific subject
  python scripts/prepare_fzhi_custom_data.py \
    --data_root /mnt/data_hdd/fzhi/output \
    --subjects 101010 \
    --val_ratio 0.15
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_JOINTS = 44   # CameraHMR gtkps: 25 OpenPose + 19 extra
NUM_BETAS = 10


def bbox_from_2d_keypoints(kp2d, vis, padding=1.2):
    """Compute center (2,) and scale from 2D keypoints. scale*200 = bbox side."""
    valid = vis > 0.5
    if valid.sum() < 2:
        cx, cy = kp2d.mean(axis=0)
        return np.array([cx, cy], dtype=np.float32), 0.5
    x, y = kp2d[valid, 0], kp2d[valid, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = (xmax - xmin) * padding
    h = (ymax - ymin) * padding
    scale = max(w, h) / 200.0
    return np.array([cx, cy], dtype=np.float32), max(scale, 0.01)


def collect_sequences(data_root, subjects, views):
    """
    Collect all (annotation_json, subject, action, scene, camera, view) tuples.
    """
    data_root = Path(data_root)
    results = []
    for subj in subjects:
        subj_dir = data_root / subj
        if not subj_dir.is_dir():
            print(f"  WARNING: subject dir not found: {subj_dir}")
            continue
        for action_dir in sorted(subj_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            for scene_dir in sorted(action_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                for cam_dir in sorted(scene_dir.iterdir()):
                    if not cam_dir.is_dir():
                        continue
                    for view in views:
                        ann_dir = cam_dir / "smpl" / view / "annotations"
                        joint_dir = cam_dir / "joint" / "npy" / "transformed"
                        if not ann_dir.exists():
                            continue
                        for ann_file in sorted(ann_dir.glob("*.json")):
                            results.append({
                                'ann_path': str(ann_file),
                                'joint_dir': str(joint_dir) if joint_dir.exists() else None,
                                'subject': subj,
                                'action': action_dir.name,
                                'scene': scene_dir.name,
                                'camera': cam_dir.name,
                                'view': view,
                            })
    return results


def process_one_sequence(seq_info, data_root):
    """Process one annotation JSON + corresponding transformed joints. Returns list of sample dicts."""
    ann_path = seq_info['ann_path']
    joint_dir = seq_info['joint_dir']
    subject = seq_info['subject']
    action = seq_info['action']
    scene = seq_info['scene']
    camera = seq_info['camera']
    view = seq_info['view']

    with open(ann_path) as f:
        frames = json.load(f)

    samples = []
    for frame in frames:
        frame_idx = frame['frame_idx']
        smpl = frame['smpl']
        cam = frame['camera']
        joints_2d = np.array(frame['joints_2d'], dtype=np.float32)   # (22, 2)
        visibility = np.array(frame['visibility'], dtype=np.float32)  # (22,)

        # --- Image path (relative to data_root) ---
        img_rel = f"{subject}/{action}/{scene}/{camera}/image/{view}/frame_{frame_idx:08d}.png"

        # --- SMPL params ---
        global_orient = np.array(smpl['global_orient'], dtype=np.float32)  # (3,)
        body_pose = np.array(smpl['body_pose'], dtype=np.float32)          # (63,)
        betas = np.array(smpl['betas'], dtype=np.float32)[:NUM_BETAS]      # (10,)
        trans = np.array(smpl['trans'], dtype=np.float32)                   # (3,)
        # pose_cam: (72,) = global_orient(3) + body_pose(63) + pad(6) = 24*3
        pose_cam = np.zeros(72, dtype=np.float32)
        pose_cam[:3] = global_orient
        pose_cam[3:66] = body_pose

        # --- Camera intrinsics (3x3) ---
        fx, fy = cam['fx'], cam['fy']
        cx, cy = cam['cx'], cam['cy']
        cam_int = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # --- Transformed 3D joints from joint/npy/transformed ---
        joints_3d_transformed = np.zeros((22, 3), dtype=np.float32)
        if joint_dir:
            npy_path = os.path.join(joint_dir, f"{frame_idx:08d}.npy")
            if os.path.exists(npy_path):
                joints_3d_transformed = np.load(npy_path).astype(np.float32)

        # --- 3D joints from annotations (camera frame) ---
        joints_3d_cam = np.array(frame['joints_3d'], dtype=np.float32)  # (22, 3)

        # --- 2D keypoints: gtkps (44, 3) ---
        gtkps = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        n = min(22, joints_2d.shape[0])
        gtkps[:n, :2] = joints_2d[:n]
        gtkps[:n, 2] = visibility[:n]

        # --- Bbox ---
        center, scale = bbox_from_2d_keypoints(joints_2d, visibility)

        # --- cam_ext: identity (SMPL params already in camera frame) ---
        cam_ext = np.eye(4, dtype=np.float32)

        samples.append({
            'imgname': img_rel,
            'center': center,
            'scale': scale,
            'pose_cam': pose_cam,
            'shape': betas,
            'cam_int': cam_int,
            'gtkps': gtkps,
            'cam_ext': cam_ext,
            'trans_cam': trans,
            'joints_3d_cam': joints_3d_cam,
            'joints_3d_transformed': joints_3d_transformed,
            # metadata for splitting
            '_subject': subject,
            '_action': action,
            '_scene': scene,
        })
    return samples


def save_npz(samples, out_path):
    """Save list of sample dicts to CameraHMR npz."""
    n = len(samples)
    if n == 0:
        print(f"  Skipping {out_path}: 0 samples")
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        imgname=np.array([s['imgname'] for s in samples], dtype=object),
        scale=np.array([s['scale'] for s in samples], dtype=np.float32),
        center=np.array([s['center'] for s in samples], dtype=np.float32),
        pose_cam=np.array([s['pose_cam'] for s in samples], dtype=np.float32),
        shape=np.array([s['shape'] for s in samples], dtype=np.float32),
        cam_int=np.array([s['cam_int'] for s in samples], dtype=np.float32),
        gtkps=np.array([s['gtkps'] for s in samples], dtype=np.float32),
        cam_ext=np.array([s['cam_ext'] for s in samples], dtype=np.float32),
        trans_cam=np.array([s['trans_cam'] for s in samples], dtype=np.float32),
        joints_3d_cam=np.array([s['joints_3d_cam'] for s in samples], dtype=np.float32),
        joints_3d_transformed=np.array([s['joints_3d_transformed'] for s in samples], dtype=np.float32),
    )
    pose_shape = np.array([s['pose_cam'] for s in samples]).shape
    print(f"  Saved {n} samples → {out_path}")
    print(f"    pose_cam: {pose_shape}, gtkps: (N,44,3), joints_3d: (N,22,3)")


def split_train_val(all_samples, val_ratio=0.1, val_actions=None, seed=42):
    """
    Split samples into train/val.
    Strategy: split by (subject, action, scene) groups to avoid data leakage.
    """
    random.seed(seed)
    # Group by (subject, action, scene)
    groups = defaultdict(list)
    for s in all_samples:
        key = (s['_subject'], s['_action'], s['_scene'])
        groups[key].append(s)

    keys = sorted(groups.keys())

    if val_actions:
        # Split by action name
        val_action_set = set(val_actions)
        val_keys = [k for k in keys if k[1] in val_action_set]
        train_keys = [k for k in keys if k[1] not in val_action_set]
    else:
        # Random split by groups
        random.shuffle(keys)
        n_val = max(1, int(len(keys) * val_ratio))
        val_keys = keys[:n_val]
        train_keys = keys[n_val:]

    train_samples = []
    for k in train_keys:
        train_samples.extend(groups[k])
    val_samples = []
    for k in val_keys:
        val_samples.extend(groups[k])

    print(f"  Split: {len(train_keys)} train groups ({len(train_samples)} samples), "
          f"{len(val_keys)} val groups ({len(val_samples)} samples)")
    return train_samples, val_samples


def main():
    ap = argparse.ArgumentParser(description="Prepare fzhi custom data for CameraHMR")
    ap.add_argument("--data_root", type=str, default="/mnt/data_hdd/fzhi/output")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output dir for npz files (default: <REPO>/data/)")
    ap.add_argument("--subjects", type=str, default=None,
                    help="Comma-separated subject IDs. Default: all.")
    ap.add_argument("--views", type=str, default="forward",
                    help="Comma-separated views (default: forward)")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of (subject,action,scene) groups for val (default: 0.1)")
    ap.add_argument("--val_actions", type=str, default=None,
                    help="Comma-separated action names to use as val set (overrides val_ratio)")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data"

    # Subject list
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    else:
        subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    views = [v.strip() for v in args.views.split(",")]

    print(f"Data root: {data_root}")
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"Views: {views}")
    print()

    # Collect sequences
    sequences = collect_sequences(data_root, subjects, views)
    print(f"Found {len(sequences)} annotation files")

    # Process all
    all_samples = []
    for i, seq in enumerate(sequences):
        if (i + 1) % 200 == 0:
            print(f"  Processing {i+1}/{len(sequences)}...")
        samples = process_one_sequence(seq, data_root)
        all_samples.extend(samples)

    if args.max_samples and len(all_samples) > args.max_samples:
        random.seed(args.seed)
        all_samples = random.sample(all_samples, args.max_samples)

    print(f"\nTotal samples: {len(all_samples)}")
    if not all_samples:
        raise SystemExit("No samples found!")

    # Split
    val_actions = None
    if args.val_actions:
        val_actions = [a.strip() for a in args.val_actions.split(",")]

    train_samples, val_samples = split_train_val(
        all_samples, val_ratio=args.val_ratio, val_actions=val_actions, seed=args.seed
    )

    # Save
    train_npz = out_dir / "training-labels" / "fzhi_custom_train.npz"
    val_npz = out_dir / "test-labels" / "fzhi_custom_val.npz"
    print()
    save_npz(train_samples, train_npz)
    save_npz(val_samples, val_npz)

    print(f"\n=== Done ===")
    print(f"Train npz: {train_npz}")
    print(f"Val npz:   {val_npz}")
    print(f"Images at: {data_root} (DATASET_FOLDERS['fzhi-custom'] should point here)")


if __name__ == "__main__":
    main()
