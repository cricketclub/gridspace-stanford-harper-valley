import os
import torch
import numpy
import random
import getpass
from copy import deepcopy
from src.systems.wav2vec import *
from src.systems.selfsup import *
from src.systems.setup import process_config
from src.utils import load_json
import pytorch_lightning as pl
import wandb


def run(config_path, caller_intent=None, gpu_device=-1):
    config = process_config(config_path,exp_name_suffix=caller_intent)
    if gpu_device >= 0: config.gpu_device = gpu_device
    # set the caller_intent
    config.data_params.caller_intent = caller_intent
    seed_everything(config.seed)
    SystemClass = globals()[config.system]
    system = SystemClass(config)

    # TODO: adjust period for saving checkpoints.
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        period=1,
    )
    wandb.init(
        project='hvb_speech', 
        entity=getpass.getuser(), 
        dir='/mnt/fs5/wumike/wandb/speech',
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
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('caller_intent', type=str, 
                        choices=['speaker_id', 'task_type', 'dialog_acts', 'sentiment'])
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()
    run(args.config, caller_intent=args.caller_intent, gpu_device=args.gpu_device)
