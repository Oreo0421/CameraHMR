# 自定义数据 + CameraHMR 预训练数据 混合训练 完整指南

## 总览

```
Step 0: 下载预训练数据 (模型 + 标签 + COCO/MPII 图像)
Step 1: 运行 prepare 脚本 → 自动生成 train.npz / val.npz
Step 2: python train.py data=fzhi_custom_train experiment=camerahmr
Step 3: 在自己数据上评估
```

---

## Step 0: 下载预训练数据（混入用）

### 0.1 一键下载模型 + 标签

```bash
cd /home/fzhi/fzt/CameraHMR
bash scripts/download_pretrain_data.sh
```

需要注册 https://camerahmr.is.tue.mpg.de/ 获取账号密码。

这个脚本下载：
- SMPL 模型 → `data/models/SMPL/SMPL_NEUTRAL.pkl`
- CameraHMR 预训练权重 → `data/pretrained-models/`
- 训练工具文件 (J_regressor 等) → `data/train-eval-utils/`
- COCO/MPII 训练标签 (npz) → `data/training-labels/coco-release.npz`, `mpii-release.npz`

### 0.2 手动下载原始图像

CameraHMR **不提供**原始图像，你需要自行下载：

| 数据集 | 下载地址 | 放到 |
|--------|---------|------|
| **COCO 2017 Train** | https://cocodataset.org/#download → "2017 Train images [118K/18GB]" | `data/training-images/COCO/images/` |
| **MPII Human Pose** | http://human-pose.mpi-inf.mpg.de/#download → "Images (12.9 GB)" | `data/training-images/MPII-pose/images/` |

> 如果你**暂时不想下载这些**，可以只用自己的数据训练。编辑 `core/configs_hydra/data/fzhi_custom_train.yaml`，把 `DATASETS_AND_RATIOS` 改为 `'fzhi-custom'`（不混入外部数据）。

---

## Step 1: 处理你的自定义数据

### 1.1 你的数据已有什么

```
/mnt/data_hdd/fzhi/output/<subject>/<action>/<scene>/<camera>/
  ├── image/<view>/frame_XXXXXXXX.png                              # 图像 (1000x1000)
  ├── smpl/<view>/annotations/<subj>_<action>_annotations.json     # SMPL + 相机 + 2D/3D joints
  ├── smpl/<view>/<subj>_<action>_smpl.npz                         # SMPL 序列
  └── joint/npy/transformed/<frame>.npy                            # 3D joints 世界坐标 (22,3)
```

annotations JSON 每帧含：`smpl{global_orient, body_pose, betas, trans}`, `camera{fx,fy,cx,cy}`, `joints_2d`, `joints_3d`, `visibility`

### 1.2 运行 prepare 脚本

```bash
cd /home/fzhi/fzt/CameraHMR

python scripts/prepare_fzhi_custom_data.py \
  --data_root /mnt/data_hdd/fzhi/output \
  --subjects "100832,100837,101010,101020,101032,101412,101883,102023,203915,204124,avatarrex_zzr" \
  --views forward \
  --val_ratio 0.1
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `--subjects` | 逗号分隔的 subject ID，选 10-11 个 |
| `--views` | 视角，默认 `forward`，可加 `forward,left,right` |
| `--val_ratio` | 按 (subject, action, scene) 分组，10% 划给 val |
| `--val_actions` | 可选：指定某些 action 名全部划给 val（覆盖 val_ratio） |

**输出：**
```
data/training-labels/fzhi_custom_train.npz   ← 训练集 (~90%)
data/test-labels/fzhi_custom_val.npz         ← 验证集 (~10%)
```

**npz 内含字段：**
- `imgname` — 相对 `/mnt/data_hdd/fzhi/output` 的图片路径
- `pose_cam` (N, 72) — SMPL 姿态 (global_orient + body_pose)
- `shape` (N, 10) — betas
- `trans_cam` (N, 3) — 平移
- `cam_int` (N, 3, 3) — 相机内参
- `gtkps` (N, 44, 3) — 2D 关键点 + visibility
- `joints_3d_cam` (N, 22, 3) — 相机坐标系 3D 关节 (从 annotations)
- `joints_3d_transformed` (N, 22, 3) — 世界坐标 3D 关节 (从 joint/npy/transformed)
- `center` (N, 2), `scale` (N,) — bbox
- `cam_ext` (N, 4, 4) — 单位矩阵 (SMPL 已在相机坐标系)

### 1.3 Train/Val 划分策略

脚本按 **(subject, action, scene)** 分组后随机划分，保证：
- **同一个人做同一动作在同一场景**的所有帧要么全在 train，要么全在 val
- 避免数据泄漏（不会出现同序列的不同帧分散在 train 和 val）

---

## Step 2: 训练

### 2.1 目录结构确认

运行前确保目录结构如下：

```
CameraHMR/
├── data/
│   ├── models/SMPL/SMPL_NEUTRAL.pkl          ← Step 0 下载
│   ├── pretrained-models/
│   │   ├── camerahmr_checkpoint_cleaned.ckpt  ← Step 0 下载
│   │   ├── cam_model_cleaned.ckpt             ← Step 0 下载
│   │   └── model_final_f05665.pkl             ← Step 0 下载
│   ├── smpl_mean_params.npz                   ← Step 0 下载
│   ├── train-eval-utils/
│   │   ├── J_regressor_extra.npy              ← Step 0 下载
│   │   ├── J_regressor_h36m.npy
│   │   ├── SMPL_to_J19.pkl
│   │   └── vitpose_backbone.pth
│   ├── training-labels/
│   │   ├── fzhi_custom_train.npz              ← Step 1 生成
│   │   ├── coco-release.npz                   ← Step 0 下载 (混入用)
│   │   └── mpii-release.npz                   ← Step 0 下载 (混入用)
│   ├── test-labels/
│   │   └── fzhi_custom_val.npz                ← Step 1 生成
│   └── training-images/
│       ├── COCO/images/                        ← 手动下载 (混入用)
│       └── MPII-pose/images/                   ← 手动下载 (混入用)
│
├── /mnt/data_hdd/fzhi/output/                  ← 你的数据 (不需要移动)
│   ├── 100832/
│   ├── 100837/
│   ├── 101010/
│   └── ...
│
├── core/configs/__init__.py                    ← 已注册 fzhi-custom, fzhi-custom-val
├── core/configs_hydra/data/fzhi_custom_train.yaml  ← 数据配置
└── scripts/
    ├── prepare_fzhi_custom_data.py             ← 数据准备
    ├── download_pretrain_data.sh               ← 下载预训练
    └── run_fzhi_custom_training.sh             ← 一键运行
```

### 2.2 训练命令

```bash
# 方法 A: 一键脚本
bash scripts/run_fzhi_custom_training.sh

# 方法 B: 手动运行
python train.py data=fzhi_custom_train experiment=camerahmr exp_name=fzhi_custom_run1
```

### 2.3 混入比例控制

编辑 `core/configs_hydra/data/fzhi_custom_train.yaml`：

```yaml
DATASETS:
  # 当前: 5x自定义 + coco + mpii ≈ 70-80% 自定义
  DATASETS_AND_RATIOS: 'fzhi-custom_fzhi-custom_fzhi-custom_fzhi-custom_fzhi-custom_coco-train_mpii-train'
```

| 想要的比例 | 改成 |
|-----------|------|
| 100% 自定义 (不混入) | `'fzhi-custom'` |
| ~80% 自定义 + 20% 预训练 | `'fzhi-custom_fzhi-custom_fzhi-custom_fzhi-custom_coco-train'` |
| ~50/50 | `'fzhi-custom_coco-train_mpii-train'` |

---

## Step 3: 评估

训练完后在你自己的 val 集上评估：

```bash
python eval.py data=fzhi_custom_train experiment=camerahmr
```

`VAL_DATASETS: fzhi-custom-val` 已在配置中设好。

---

## 快速开始 (最小配置，不混入外部数据)

如果你暂时不想下载 COCO/MPII，可以只用自己的数据：

```bash
# 1. 只下载必需的模型文件
bash scripts/download_pretrain_data.sh   # 跳过 COCO/MPII 图像下载

# 2. 处理你的数据
python scripts/prepare_fzhi_custom_data.py \
  --data_root /mnt/data_hdd/fzhi/output \
  --subjects "101010" \
  --val_ratio 0.15

# 3. 编辑配置，只用自定义数据
#    core/configs_hydra/data/fzhi_custom_train.yaml:
#    DATASETS_AND_RATIOS: 'fzhi-custom'

# 4. 训练
python train.py data=fzhi_custom_train experiment=camerahmr exp_name=fzhi_only_run1
```
