import argparse
import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.text8_module import Text8Module
from datasets import get_dataset
from utils import load_config

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    pl.seed_everything(config.train.seed)

    # Data
    train_set, valid_set = get_dataset(config.datasets, return_test=False)

    # Dataloader
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=config.train.batch_size, shuffle=False)

    # Model
    pl_module = Text8Module(**config)
    trainer = pl.Trainer(
        accelerator='auto',
        logger=TensorBoardLogger(
            args.logdir,
            name=args.savename,
            version=0,
            default_hp_metric=False,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(args.logdir, args.savename),
                save_top_k=3,
                save_last=True,
                monitor='val_loss',
                filename='{epoch}-{step}',
            ),
        ],
        max_epochs=-1,
        max_steps=config.train.max_steps,
        limit_val_batches=config.train.limit_val_batches,
        val_check_interval=config.train.val_check_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.train.log_every_n_steps,
        enable_progress_bar=True,
        gradient_clip_val=config.train.gradient_clip_val,
        precision='bf16-mixed',
    )
    trainer.fit(pl_module, train_loader, valid_loader, ckpt_path=args.resume)
