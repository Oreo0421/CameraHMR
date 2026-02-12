#!/bin/bash
# =============================================================================
# 下载 CameraHMR 混入训练所需的预训练数据（标签 + 模型 + 工具文件）
#
# 需要的数据分三部分：
#   1) 训练标签 (npz): COCO, MPII 的 SMPL 拟合参数  [~几百MB]
#   2) 训练图像: COCO, MPII 原始图像               [需自行下载]
#   3) 模型文件 + 工具文件: SMPL模型, 预训练权重    [必须]
#
# 注册地址:
#   - CameraHMR: https://camerahmr.is.tue.mpg.de/
#   - (可选) BEDLAM: https://bedlam.is.tue.mpg.de/ (如需 BEDLAM 数据)
# =============================================================================

set -e
cd "$(dirname "$0")/.."  # cd to repo root

urle () {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

echo "========================================="
echo " Step 1: CameraHMR 模型 + 工具文件"
echo "========================================="
echo "注册地址: https://camerahmr.is.tue.mpg.de/"
read -p "Username (CameraHMR): " username
read -p "Password (CameraHMR): " password
username=$(urle $username)
password=$(urle $password)

# SMPL 模型
mkdir -p data/models/SMPL
echo "下载 SMPL 模型..."
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=SMPL.zip' \
    -O './data/models/SMPL.zip' --no-check-certificate --continue
unzip -o data/models/SMPL.zip -d data/models/

# 预训练权重
mkdir -p data/pretrained-models
echo "下载预训练权重..."
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=camerahmr_checkpoint_cleaned.ckpt' \
    -O './data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=cam_model_cleaned.ckpt' \
    -O './data/pretrained-models/cam_model_cleaned.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=model_final_f05665.pkl' \
    -O './data/pretrained-models/model_final_f05665.pkl' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=smpl_mean_params.npz' \
    -O './data/smpl_mean_params.npz' --no-check-certificate --continue

# 训练/评估工具文件 (J_regressor等)
echo "下载工具文件..."
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=train-eval-utils.zip' \
    -O './data/train-eval-utils.zip' --no-check-certificate --continue
unzip -o data/train-eval-utils.zip -d data/

echo ""
echo "========================================="
echo " Step 2: 预训练数据 标签 (COCO/MPII)"
echo "========================================="
# 训练标签 (SMPL 拟合参数)
mkdir -p data/training-labels
echo "下载 COCO/MPII 训练标签..."
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=coco-release.npz' \
    -O './data/training-labels/coco-release.npz' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=mpii-release.npz' \
    -O './data/training-labels/mpii-release.npz' --no-check-certificate --continue

# 测试标签
echo "下载测试标签..."
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=test-labels.zip' \
    -O './data/test-labels.zip' --no-check-certificate --continue
unzip -o data/test-labels.zip -d data/

echo ""
echo "========================================="
echo " Step 3: 需要你自己下载的原始图像"
echo "========================================="
echo ""
echo "CameraHMR 不提供原始图像，你需要自行下载并放到以下目录："
echo ""
echo "  COCO 2017 train images:"
echo "    下载: https://cocodataset.org/#download → 2017 Train images (train2017.zip)"
echo "    解压后需得到: data/training-images/COCO/images/train2017/*.jpg"
echo "    (npz 里是 2014 路径，代码会自动回退到 train2017/ 读取)"
echo ""
echo "  MPII Human Pose:"
echo "    下载: http://human-pose.mpi-inf.mpg.de/#download → Images (12.9 GB)"
echo "    解压到: data/training-images/MPII-pose/images/"
echo ""
echo "========================================="
echo " Done! 目录结构应为:"
echo "========================================="
echo ""
echo "  data/"
echo "  ├── models/SMPL/SMPL_NEUTRAL.pkl"
echo "  ├── pretrained-models/"
echo "  │   ├── camerahmr_checkpoint_cleaned.ckpt"
echo "  │   ├── cam_model_cleaned.ckpt"
echo "  │   └── model_final_f05665.pkl"
echo "  ├── smpl_mean_params.npz"
echo "  ├── train-eval-utils/"
echo "  │   ├── J_regressor_extra.npy"
echo "  │   ├── J_regressor_h36m.npy"
echo "  │   ├── SMPL_to_J19.pkl"
echo "  │   └── vitpose_backbone.pth"
echo "  ├── training-labels/"
echo "  │   ├── coco-release.npz        ← 预训练标签"
echo "  │   ├── mpii-release.npz        ← 预训练标签"
echo "  │   └── fzhi_custom_train.npz   ← 你的数据 (prepare脚本生成)"
echo "  ├── test-labels/"
echo "  │   └── fzhi_custom_val.npz     ← 你的数据 (prepare脚本生成)"
echo "  └── training-images/"
echo "      ├── COCO/images/             ← 你自己下载"
echo "      └── MPII-pose/images/        ← 你自己下载"
