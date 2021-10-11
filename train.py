#!/usr/bin/env python
# coding: utf-8

import torch, torchaudio
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
logger = logging.getLogger(__name__)

import os
import keepsake
from keepsake.pl_callback import KeepsakeCallback 


class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self, path: Path = Path('data/ESC-50'), 
                 sample_rate: int = 8000,
                 folds = [1]):
        # Load CSV & initialize all torchaudio.transforms:
        # Resample --> MelSpectrogram --> AmplitudeToDB
        self.path = path
        self.csv = pd.read_csv(path / Path('meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        self.resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        
    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / 'audio' / row['filename'])
        label = row['target']
        xb = self.db(
            self.melspec(
                self.resample(wav)
            )
        )
        return xb, label
        
    def __len__(self):
        # Returns length
        return len(self.csv)

class AudioNet(pl.LightningModule):
    
    def __init__(self, hparams: DictConfig):
        super().__init__()

        # Fundamental, this will save the hyper-parameters
        # in a way that is meaningful to PyTorch Lightning.
        self.save_hyperparameters(hparams) 

        self.conv1 = nn.Conv2d(1, hparams.base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(hparams.base_filters)
        self.conv2 = nn.Conv2d(hparams.base_filters, hparams.base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hparams.base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(hparams.base_filters, hparams.base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hparams.base_filters * 2)
        self.conv4 = nn.Conv2d(hparams.base_filters * 2, hparams.base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(hparams.base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(hparams.base_filters * 4, hparams.n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x
    
    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = functional.accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer


@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.
    
    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))
    
    
    
    # We recover the original path of the dataset:
    path = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.path)

    # Load data
    train_data = ESC50Dataset(path=path, folds=cfg.data.train_folds)
    val_data = ESC50Dataset(path=path, folds=cfg.data.val_folds)
    test_data = ESC50Dataset(path=path, folds=cfg.data.test_folds)

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size)

    pl.seed_everything(cfg.seed)

    # Initialize the network
    audionet = AudioNet(cfg.model)
    
    os.chdir("../../..")
    
    print(cfg)
    print(type(cfg))
 
    
    trainer = pl.Trainer(**cfg.trainer, checkpoint_callback=False, callbacks=[KeepsakeCallback(primary_metric=("train_loss", "minimize"), period=1,
    params =  {"base_filters": cfg['model']["base_filters"], 'n_classes': cfg['model']['n_classes'], 'learning_rate': cfg['model']['optimizer']['lr']})])
    trainer.fit(audionet, train_loader, val_loader)

if __name__ == "__main__":
    train()
