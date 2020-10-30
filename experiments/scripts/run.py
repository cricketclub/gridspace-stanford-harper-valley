import os
import torch
import numpy
import random
from copy import deepcopy
from src.systems.recognition import (
    CTC_System,
    LAS_System,
    MTL_System,
)
from src.systems.wav2vec import Wav2Vec_System
from src.systems.selfsup import (
    InstDiscSystem,
    LocalAggSystem,
    SimCLRSystem,
    TransferLibriSpeechSystem,
    TransferHarperValleySystem,
)
from src.systems.setup import process_config
from src.utils import load_json
import pytorch_lightning as pl
import wandb


def run(config_path, gpu_device=-1):
    config = process_config(config_path)
    if gpu_device >= 0: config.gpu_device = gpu_device
    seed_everything(config.seed)
    SystemClass = globals()[config.system]
    system = SystemClass(config)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        period=1,
    )
    wandb.init(
        project='speech', 
        entity='testuser', 
        name=config.exp_name, 
        config=config, 
        sync_tensorboard=True,
    )
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=([config.gpu_device] if config.cuda else None),
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.continue_from_checkpoint,
    )
    trainer.fit(system)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()
    run(args.config, gpu_device=args.gpu_device)
