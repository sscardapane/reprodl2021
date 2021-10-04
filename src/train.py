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

import pickle
import sys
import os
import yaml
from prepare_data import ESC50Dataset


class AudioNet(pl.LightningModule):

    def __init__(self, n_classes = 50, base_filters = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(base_filters, base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 2)
        self.conv4 = nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(base_filters * 4, n_classes)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train(train_data, val_data, test_data, n_classes=20, base_filters=16,\
        batch_size=8, max_epochs=5):
    # input: ESC50Dataset objects train_data, val_data, test_data
    # This is the main training function requested by the exercise.
    # We use folds 1,2,3 for training, 4 for validation, 5 for testing.

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    pl.seed_everything(0)

    # Initialize the network
    audionet = AudioNet(n_classes, base_filters)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(audionet, train_loader, val_loader)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    train_data_file = os.path.join(sys.argv[1], "train.pickle")
    val_data_file = os.path.join(sys.argv[1], "val.pickle")
    test_data_file = os.path.join(sys.argv[1], "test.pickle")

    with open(train_data_file, 'rb') as train_file:
        train_data = pickle.load(train_file)

    with open(val_data_file, 'rb') as val_file:
        val_data = pickle.load(val_file)

    with open(test_data_file, 'rb') as test_file:
        test_data = pickle.load(test_file)

    params = yaml.safe_load(open("../params.yaml"))
    params_audio_net = params["audio_net"]
    params_data_loader = params["data_loader"]
    params_training = params["training"]

    n_classes = params_audio_net["n_classes"]
    base_filters = params_audio_net["base_filters"]

    batch_size = params_data_loader["batch_size"]
    max_epochs = params_training["max_epochs"]

    train(train_data, val_data, test_data, n_classes, base_filters, batch_size, max_epochs)
