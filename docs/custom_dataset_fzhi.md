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
| **COCO 2017 Train** | https://cocodataset.org/#download → "2017 Train images [118K/18GB]" | 解压后需为 `data/training-images/COCO/images/train2017/*.jpg`（代码会对 npz 里的 2014 路径自动回退到 `train2017/`） |
| **COCO 2014 Train** | 见下方一键下载 | 解压后为 `data/training-images/COCO/images/train2014/*.jpg`（与 npz 内路径一致） |
| **MPII Human Pose** | http://human-pose.mpi-inf.mpg.de/#download → "Images (12.9 GB)" | `data/training-images/MPII-pose/images/`（npz 内路径若不同则可能需改代码或只用自己的数据） |

**COCO 2014 一键下载（约 13.5GB）：**
```bash
cd /home/fzhi/fzt/CameraHMR
mkdir -p data/training-images/COCO/images
wget -c http://images.cocodataset.org/zips/train2014.zip -O data/training-images/COCO/images/train2014.zip
unzip -o data/training-images/COCO/images/train2014.zip -d data/training-images/COCO/images/
# 解压后为 data/training-images/COCO/images/train2014/*.jpg
rm data/training-images/COCO/images/train2014.zip   # 可选，省空间
```

> 如果你**暂时不想下载这些**，可以只用自己的数据训练。编辑 `core/configs_hydra/data/fzhi_custom_train.yaml`，把 `DATASETS_AND_RATIOS` 改为 `'fzhi-custom'`（不混入外部数据）。

---

## 重新混入官方训练集（COCO/MPII）

1. **确保图片路径正确**
   - **COCO 2014**（推荐）：解压到 `data/training-images/COCO/images/train2014/`，文件名为 `COCO_train2014_*.jpg`
   - COCO 2017：`data/training-images/COCO/images/train2017/*.jpg`（代码会从 npz 的 2014 路径回退到 train2017）
   - MPII：`data/training-images/MPII-pose/images/` 或 `MPII-pose/076265256.jpg`

2. **混合训练配置（已默认开启 + 缩短时间）**
   - `fzhi_custom_train.yaml` 已设为：`DATASETS_AND_RATIOS: 'fzhi-custom_fzhi-custom_fzhi-custom_fzhi-custom_coco-train'`（约 80% 自定义 + 20% COCO）
   - `TOTAL_STEPS: 60_000`（约 1～2 天）；`VAL_STEPS: 8000`、`limit_val_batches: 0.25` 减少验证时间
   - 要加 MPII：在末尾加 `_mpii-train`。要只用自己的数据：改为 `'fzhi-custom'`。

3. **训练**
   ```bash
   python train.py data=fzhi_custom_train experiment=camerahmr exp_name=xxx
   ```

### 看每步 loss、预计训练时间

- **终端进度条**：每个 step 的 `train/loss` 会显示在进度条上；进度条会显示当前 step、总 step 和 **ETA（预计剩余时间）**。  
- **打印频率**：由 `GENERAL.LOG_STEPS` 控制（fzhi_custom_train 默认 50），即每 50 step 打一次 loss；改小（如 20）可更频繁看到数值。  
- **完整曲线**：用 TensorBoard 看 `train/loss` 随 step 的变化：
  ```bash
  tensorboard --logdir=ckpt/train/runs
  ```
  在浏览器打开提示的地址即可。  
- **大概要训多久**：跑几百个 step 后看进度条上的 **it/s 或 s/it**，用 `总 step 数 × 每 step 时间` 估算；进度条 ETA 会随训练动态更新。

### 为什么一个 epoch 很慢？总共几个 epoch？

- **按 step 训练，没有“总 epoch 数”**：代码用的是 `max_steps`（即 `GENERAL.TOTAL_STEPS`），训练会一直跑到总步数为止，不会“跑 N 个 epoch 就停”。  
- **一个 epoch** = 把你的训练集从头到尾过一遍 = `ceil(训练样本数 / BATCH_SIZE)` 个 step。例如 10 万张图、batch_size=64，一个 epoch ≈ 1562 step。  
- **一个 epoch 要几小时**：要么数据量很大（每 epoch 的 step 多），要么每 step 较慢（读图、ViT 前向等）。  
- **只用自己数据时**：`fzhi_custom_train` 里已把 **TOTAL_STEPS 设为 100_000**（10 万步），这样不会按默认的 100 万步训很久。若想快速试跑可改为 `50_000`；要训满再改回 `1_000_000`。  
- **加速建议**：显存允许可增大 `TRAIN.BATCH_SIZE`（如 128），每 epoch 的 step 数会减半；或适当增大 `GENERAL.NUM_WORKERS`（如 8～16）减轻读图瓶颈。

### 修改配置后重新续跑

1. **改配置**  
   编辑 `core/configs_hydra/data/fzhi_custom_train.yaml`（或命令行覆盖，如 `general.val_steps=5000`）。

2. **用同一实验名续跑**  
   第一次跑时用了哪个 `exp_name`，续跑时**必须写同一个**，这样日志和 checkpoint 目录一致，会自动从该目录下的 `checkpoints/last.ckpt` 恢复：
   ```bash
   cd /home/fzhi/fzt/CameraHMR
   python train.py data=fzhi_custom_train experiment=camerahmr exp_name=fzhi_only
   ```
   若第一次是 `exp_name=my_run`，就继续用 `exp_name=my_run`。

3. **或显式指定 checkpoint 路径**  
   想从别的目录的某次运行续跑，或指定某个 ckpt 文件时：
   ```bash
   python train.py data=fzhi_custom_train experiment=camerahmr ckpt_path=logs/train/runs/fzhi_only/checkpoints/last.ckpt
   ```

4. **输出目录**  
   日志和 checkpoint 在：`logs/train/runs/<exp_name>/`，其中 `checkpoints/last.ckpt` 为最新权重，用于续跑。

### 在真实数据上验证不同 step 的 checkpoint（选多少步用于 test）

在**真实 val 集**（fzhi-custom-val）上分别跑 50k、60k、70k 等 checkpoint，看 PA-MPJPE/MPJPE 等指标，选表现最好的步数。

1. **复制带等号文件名的 ckpt（避免 Hydra 解析错误）**  
   路径里的 `epoch=4-step=70000.ckpt` 不能直接当命令行参数，先复制成无等号文件名：
   ```bash
   CDIR=/mnt/data_hdd/fzhi/CameraHMR/ckpt/train/runs/mixed_coco/checkpoints
   cp "$CDIR/epoch=3-step=50000.ckpt" "$CDIR/step50000.ckpt"
   cp "$CDIR/epoch=4-step=60000.ckpt" "$CDIR/step60000.ckpt"
   cp "$CDIR/epoch=4-step=70000.ckpt" "$CDIR/step70000.ckpt"
   ```

2. **在真实 val 上跑 eval（指定 data + ckpt_path）**  
   ```bash
   cd /home/fzhi/fzt/CameraHMR
   # 50k
   python eval.py data=fzhi_custom_eval ckpt_path=/mnt/data_hdd/fzhi/CameraHMR/ckpt/train/runs/mixed_coco/checkpoints/step50000.ckpt
   # 60k
   python eval.py data=fzhi_custom_eval ckpt_path=/mnt/data_hdd/fzhi/CameraHMR/ckpt/train/runs/mixed_coco/checkpoints/step60000.ckpt
   # 70k
   python eval.py data=fzhi_custom_eval ckpt_path=/mnt/data_hdd/fzhi/CameraHMR/ckpt/train/runs/mixed_coco/checkpoints/step70000.ckpt
   ```

3. **看终端输出的 PA-MPJPE、MPJPE、PVE**，对比哪个 step 在你真实数据上最好，就用该 step 的 ckpt 做后续 test/部署。

### 在真实数据集 own_omni action16 上验证准确率

你的真实数据在 **rawframes** + **annotations_final**（COCO 格式 2D 关键点），无 3D GT，只能看 **2D 关键点准确率**（avgpck_0.05 / avgpck_0.1）。

1. **生成评估用 npz（只需跑一次）**
   ```bash
   cd /home/fzhi/fzt/CameraHMR
   python scripts/prepare_own_omni_eval.py \
     --rawframes /mnt/dst_datasets/own_omni_dataset/action16_2022/rawframes \
     --annotations /mnt/dst_datasets/own_omni_dataset/action16_2022/annotations_final \
     --out_npz data/test-labels/own_omni_action16_val.npz
   ```

2. **用某个 checkpoint 在真实数据上跑 eval**
   ```bash
   python eval.py data=own_omni_eval ckpt_path=/mnt/data_hdd/fzhi/CameraHMR/ckpt/train/runs/mixed_coco/checkpoints/step60000.ckpt
   ```
   终端里会打印 **avgpck_0.05**、**avgpck_0.1**（2D PCK），数值越高表示在你真实数据上 2D 关键点越准。不会报 PA-MPJPE/MPJPE（无 3D 真值）。

3. **换不同 step 的 ckpt** 重复第 2 步，对比哪个 step 的 avgpck 最高，就用该 ckpt 做部署。

---

## 加入其他数据集（如 PoseTrack、自建数据等）

只要数据能做成 **CameraHMR 同款 npz**，就可以和 fzhi-custom、coco-train、mpii-train 一起混训。

### npz 格式要求（与现有训练集一致）

- **imgname**：相对「图像根目录」的路径，如 `train2017/0000001.jpg` 或 `subject/action/frame.png`
- **scale** (N,)：每人 bbox 的 scale（与代码里 scale*200 一致）
- **center** (N, 2)：bbox 中心 (x, y)
- **pose_cam** (N, 72)：SMPL 姿态，global_orient(3) + body_pose(69)
- **shape** (N, 10)：SMPL betas
- **cam_int** (N, 3, 3)：相机内参 3×3
- **gtkps** (N, 44, 3)：2D 关键点 (x, y, visibility)
- 可选：**cam_ext** (N, 4, 4)、**trans_cam** (N, 3)

### 操作步骤

1. **准备数据**  
   把新数据集做成上述 npz，并把所有图片放到一个「图像根目录」下，npz 里的 `imgname` 相对该根目录。

2. **在配置里注册**
   - 打开 `core/configs/__init__.py`
   - 在 `DATASET_FOLDERS` 里加一项，例如：
     ```python
     'posefes-train': '/path/to/your/posefes/images/root',  # 图像根目录
     ```
   - 在 `DATASET_FILES[1]`（训练用）里加一项：
     ```python
     'posefes-train': os.path.join(base_dir, 'data/training-labels/posefes_train.npz'),
     ```
   - npz 实际路径和名字按你本地放的位置改。

3. **加入混合训练**
   - 编辑 `core/configs_hydra/data/fzhi_custom_train.yaml`
   - 在 `DATASETS_AND_RATIOS` 里加上新数据集名（下划线分隔），例如：
     ```yaml
     DATASETS_AND_RATIOS: 'fzhi-custom_fzhi-custom_coco-train_mpii-train_posefes-train'
     ```
   - 比例仍通过「重复写名字」调节：名字出现越多，该数据集被采到的次数越多。

4. **注意**
   - 若新数据集没有官方 SMPL 拟合，需要自己用 HMR/VIBE/CLIFF 等先拟合出 pose/shape，再按上面格式写进 npz。
   - **PoseFES** 若指某个具体数据集，需要其提供 SMPL 或 3D 关节标注，再转成上述 npz；若只有 2D 关键点，需要先跑一遍 SMPL 拟合流程再做成 npz。

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
