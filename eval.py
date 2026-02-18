from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from yacs.config import CfgNode
from core.configs import dataset_config
from core.datasets import DataModule
from core.cam_model.fl_net import FLNet

from core.utils.pylogger import get_pylogger
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)
import torch
torch.set_float32_matmul_precision('medium')
torch.manual_seed(0)
warnings.filterwarnings("ignore", category=UserWarning)


def eval(cfg: DictConfig) -> Tuple[dict, dict]:
    # Load dataset config
    dataset_cfg = dataset_config()
    model_type = cfg.get('model_type', 'smpl')

    datamodule = DataModule(cfg, dataset_cfg)
    # Setup model：若传了 ckpt_path 则用该 checkpoint（自己训的）；否则用预训练默认路径
    # eval 时若只改 data 未带 experiment，需补上 MODEL（这里默认 smpl）
    model_type = cfg.get('MODEL', {}).get('TYPE', 'smpl')
    if model_type == 'smpl':
        from core.constants import CHECKPOINT_PATH
        from core.camerahmr_trainer_smpl import CameraHMR
        checkpoint_path = os.environ.get('CKPT_PATH') or cfg.get('ckpt_path') or CHECKPOINT_PATH
    elif model_type == 'smplx':
        from core.constants import CHECKPOINT_PATH_SMPLX
        from core.camerahmr_trainer_smplx import CameraHMR
        checkpoint_path = os.environ.get('CKPT_PATH') or cfg.get('ckpt_path') or CHECKPOINT_PATH_SMPLX
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    log.info(f"Loading checkpoint: {checkpoint_path}")
    model = CameraHMR.load_from_checkpoint(checkpoint_path, strict=False)

    # 把 Hydra cfg 中的 EXTRA 注入到 model.cfg（checkpoint 里没有这个字段）
    extra_cfg = cfg.get('EXTRA', None)
    if extra_cfg is not None:
        extra_dict = OmegaConf.to_container(extra_cfg, resolve=True)
        OmegaConf.set_struct(model.cfg, False)
        model.cfg.EXTRA = extra_dict
        OmegaConf.set_struct(model.cfg, True)
        log.info(f"Injected EXTRA config: SAVE_VIS={model.cfg.EXTRA.SAVE_VIS}, VIS_OUT_DIR={model.cfg.EXTRA.VIS_OUT_DIR}")

    # 只有用官方预训练 ckpt 时才单独加载 cam_model；自己训的 ckpt 里已含 cam_model
    if not os.environ.get('CKPT_PATH') and cfg.get('ckpt_path') is None:
        from core.constants import CAM_MODEL_CKPT
        cam_model_checkpoint = torch.load(CAM_MODEL_CKPT)['state_dict']
        model.cam_model.load_state_dict(cam_model_checkpoint)

    trainer = pl.Trainer()

    trainer.test(model, datamodule=datamodule)
    log.info("Fitting done")


@hydra.main(version_base="1.2", config_path=str(root/"core/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    eval(cfg)


if __name__ == "__main__":
    main()