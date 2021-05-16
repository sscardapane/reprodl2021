import torch, torchaudio
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


import hydra
from omegaconf import DictConfig, OmegaConf


import logging
logger = logging.getLogger(__name__)




class AudioNet(nn.Module):
    
    def __init__(self, hparams: DictConfig):
        super().__init__()

        #self.save_hyperparameters(hparams)

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
    '''
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
                    acc = pl.metrics.functional.accuracy(y_hat, y)
                    self.log('val_acc', acc, on_epoch=True, prog_bar=True)
                    return acc
                    
                def configure_optimizers(self):
                    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
                    return optimizer
            '''