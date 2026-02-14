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
    if cfg.MODEL.TYPE == 'smpl':
        from core.constants import CHECKPOINT_PATH
        from core.camerahmr_trainer_smpl import CameraHMR
        checkpoint_path = cfg.get('ckpt_path') or CHECKPOINT_PATH
    elif cfg.MODEL.TYPE  == 'smplx':
        from core.constants import CHECKPOINT_PATH_SMPLX
        from core.camerahmr_trainer_smplx import CameraHMR
        checkpoint_path = cfg.get('ckpt_path') or CHECKPOINT_PATH_SMPLX
    else:
        raise ValueError(f"Unknown model type: {cfg.MODEL.TYPE}")

    log.info(f"Loading checkpoint: {checkpoint_path}")
    model = CameraHMR.load_from_checkpoint(checkpoint_path, strict=False)

    # 只有用官方预训练 ckpt 时才单独加载 cam_model；自己训的 ckpt 里已含 cam_model
    if cfg.get('ckpt_path') is None:
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