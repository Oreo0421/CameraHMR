#!/bin/bash
# =============================================================================
# 完整训练流程: 自定义数据 + CameraHMR 预训练数据混合训练
#
# 整个流程 4 步:
#   Step 0: 下载预训练数据 (模型+标签+图像)
#   Step 1: 处理你的自定义数据 → 生成 train/val npz
#   Step 2: 训练
#   Step 3: 评估
# =============================================================================

set -e
cd "$(dirname "$0")/.."  # cd to repo root
echo "Working dir: $(pwd)"

# =====================
# 配置区 - 按需修改
# =====================
DATA_ROOT="/mnt/data_hdd/fzhi/output"

# 选 10-11 个 subject 用于训练 (逗号分隔)
SUBJECTS="100832,100837,101010,101020,101032,101412,101883,102023,203915,204124,avatarrex_zzr"

# 只用 forward 视角 (也可以加 left,right 等)
VIEWS="forward"

# Val 比例 (按 subject+action+scene 分组, 10% 用于 val)
VAL_RATIO=0.1

# 实验名
EXP_NAME="fzhi_custom_run1"

echo ""
echo "============================================"
echo " Step 0: 检查预训练数据是否已下载"
echo "============================================"
if [ ! -f "data/models/SMPL/SMPL_NEUTRAL.pkl" ] || \
   [ ! -f "data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt" ] || \
   [ ! -f "data/smpl_mean_params.npz" ] || \
   [ ! -f "data/train-eval-utils/vitpose_backbone.pth" ]; then
    echo ""
    echo "预训练数据未完整下载! 请先运行:"
    echo "  bash scripts/download_pretrain_data.sh"
    echo ""
    echo "然后下载 COCO/MPII 原始图像放到:"
    echo "  data/training-images/COCO/images/   (COCO 2017 train)"
    echo "  data/training-images/MPII-pose/images/ (MPII pose images)"
    echo ""
    exit 1
fi
echo "预训练数据已就绪 ✓"

# 检查混入数据标签
if [ ! -f "data/training-labels/coco-release.npz" ]; then
    echo "WARNING: data/training-labels/coco-release.npz 不存在"
    echo "  如果你不需要混入 COCO 数据, 请编辑"
    echo "  core/configs_hydra/data/fzhi_custom_train.yaml"
    echo "  把 DATASETS_AND_RATIOS 改为只有 fzhi-custom"
fi

echo ""
echo "============================================"
echo " Step 1: 处理自定义数据 → train/val npz"
echo "============================================"
echo "Data root: $DATA_ROOT"
echo "Subjects: $SUBJECTS"
echo "Views: $VIEWS"
echo "Val ratio: $VAL_RATIO"
echo ""

python scripts/prepare_fzhi_custom_data.py \
    --data_root "$DATA_ROOT" \
    --subjects "$SUBJECTS" \
    --views "$VIEWS" \
    --val_ratio $VAL_RATIO

echo ""
echo "检查生成的文件:"
ls -lh data/training-labels/fzhi_custom_train.npz 2>/dev/null || echo "  ERROR: train npz not found!"
ls -lh data/test-labels/fzhi_custom_val.npz 2>/dev/null || echo "  ERROR: val npz not found!"

echo ""
echo "============================================"
echo " Step 2: 开始训练"
echo "============================================"
echo "Experiment: $EXP_NAME"
echo "Config: data=fzhi_custom_train experiment=camerahmr"
echo ""

python train.py \
    data=fzhi_custom_train \
    experiment=camerahmr \
    exp_name=$EXP_NAME

echo ""
echo "============================================"
echo " Done! 训练完成"
echo "============================================"
echo "Checkpoints 在: logs/train/$EXP_NAME/checkpoints/"
echo ""
echo "评估命令:"
echo "  python eval.py data=fzhi_custom_train experiment=camerahmr"
