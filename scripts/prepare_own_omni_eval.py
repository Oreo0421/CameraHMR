#!/usr/bin/env python3
"""
将 own_omni_dataset/action16_2022 的 COCO 格式标注转为 CameraHMR 评估用 npz。
- 输入: rawframes/, annotations_final/*.json
- 输出: data/test-labels/own_omni_action16_val.npz
真实数据只有 2D 关键点，eval 时只会报 2D 准确率（avgpck_0.05 / avgpck_0.1），无 3D 指标。

用法:
  python scripts/prepare_own_omni_eval.py \
    --rawframes /mnt/dst_datasets/own_omni_dataset/action16_2022/rawframes \
    --annotations /mnt/dst_datasets/own_omni_dataset/action16_2022/annotations_final \
    --out_npz data/test-labels/own_omni_action16_val.npz
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

# CameraHMR gtkps 需要 44 个关键点；COCO 只有 17，其余填 0
NUM_JOINTS = 44
COCO_KEYPOINTS = 17


def bbox_to_center_scale(bbox, padding=1.2):
    """bbox [x,y,w,h] -> center (2,), scale (float, scale*200=边长)."""
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * padding
    scale = side / 200.0
    return np.array([cx, cy], dtype=np.float32), max(scale, 0.01)


def coco_kps_to_gtkps(keypoints, vis_thresh=0.5):
    """COCO 17*3 (x,y,v) -> (44, 3), 前 17 为 COCO，其余为 0."""
    kp = np.array(keypoints, dtype=np.float32).reshape(-1, 3)
    # 只取前 17
    kp = kp[:COCO_KEYPOINTS].copy()
    # visibility: 0=未标注, 1=遮挡, 2=可见 -> 转为 0/1
    kp[:, 2] = (kp[:, 2] >= 2).astype(np.float32)
    # 补到 44
    out = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    out[:kp.shape[0]] = kp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawframes", type=str, default="/mnt/dst_datasets/own_omni_dataset/action16_2022/rawframes")
    ap.add_argument("--annotations", type=str, default="/mnt/dst_datasets/own_omni_dataset/action16_2022/annotations_final")
    ap.add_argument("--out_npz", type=str, default="data/test-labels/own_omni_action16_val.npz")
    args = ap.parse_args()

    rawframes = Path(args.rawframes)
    ann_dir = Path(args.annotations)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    imgname_list = []
    center_list = []
    scale_list = []
    gtkps_list = []
    # 无 3D GT，给 pose/shape/cam_int 占位
    pose_cam_list = []
    shape_list = []
    cam_int_list = []

    json_files = sorted(ann_dir.glob("*.json"))
    for jpath in json_files:
        # 例如 brooming_aileen_bedroom.json -> action=brooming, subdir=aileen_bedroom
        stem = jpath.stem  # brooming_aileen_bedroom
        parts = stem.split("_", 2)  # ['brooming', 'aileen', 'bedroom'] or ['brooming', 'aileen_bedroom']
        if len(parts) < 2:
            continue
        action = parts[0]
        subdir = "_".join(parts[1:])  # aileen_bedroom
        rel_dir = f"{action}/{subdir}"

        with open(jpath) as f:
            data = json.load(f)
        images = {im["id"]: im for im in data["images"]}
        for ann in data.get("annotations", []):
            image_id = ann["image_id"]
            if image_id not in images:
                continue
            im = images[image_id]
            file_name = im["file_name"]
            w, h = im["width"], im["height"]
            # 相对 rawframes 的路径
            rel_path = f"{rel_dir}/{file_name}"
            abs_path = rawframes / rel_path
            if not abs_path.exists():
                continue
            bbox = ann.get("bbox", [0, 0, w, h])
            keypoints = ann.get("keypoints", [0] * (17 * 3))
            center, scale = bbox_to_center_scale(bbox)
            gtkps = coco_kps_to_gtkps(keypoints)

            imgname_list.append(rel_path)
            center_list.append(center)
            scale_list.append(scale)
            gtkps_list.append(gtkps)
            pose_cam_list.append(np.zeros(24 * 3, dtype=np.float32))
            shape_list.append(np.zeros(10, dtype=np.float32))
            # 假内参，仅用于投影
            fl = max(w, h) * 1.2
            cam_int_list.append(np.array([[fl, 0, w / 2.0], [0, fl, h / 2.0], [0, 0, 1]], dtype=np.float32))

    if not imgname_list:
        raise SystemExit("No samples found. Check paths and json format.")

    imgname_list = np.array(imgname_list, dtype=object)
    center_list = np.stack(center_list, axis=0)
    scale_list = np.array(scale_list, dtype=np.float32)
    gtkps_list = np.stack(gtkps_list, axis=0)
    pose_cam_list = np.stack(pose_cam_list, axis=0)
    shape_list = np.stack(shape_list, axis=0)
    cam_int_list = np.stack(cam_int_list, axis=0)

    np.savez(
        out_npz,
        imgname=imgname_list,
        center=center_list,
        scale=scale_list,
        gtkps=gtkps_list,
        pose_cam=pose_cam_list,
        shape=shape_list,
        cam_int=cam_int_list,
    )
    print(f"Saved {len(imgname_list)} samples -> {out_npz}")
    print("Eval 时只会报 2D 准确率 (avgpck_0.05 / avgpck_0.1)，无 3D GT 故无 PA-MPJPE/MPJPE。")


if __name__ == "__main__":
    main()
