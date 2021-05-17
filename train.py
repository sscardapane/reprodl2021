# !/usr/bin/env python
# coding: utf-8

import torch, torchaudio
from torch import nn
from torch.nn import functional as F

from ray.tune.integration.pytorch_lightning import TuneReportCallback

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional

from pathlib import Path
import pandas as pd
from functools import partial

import logging

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

logger = logging.getLogger(__name__)


class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50

    def __init__(self, path: Path = Path('data/ESC-50'),
                 sample_rate: int = 8000,
                 folds=[1]):
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

    def __init__(self, base_filters, n_classes, lr):
        super().__init__()
        self.lr = lr

        # Fundamental, this will save the hyper-parameters
        # in a way that is meaningful to PyTorch Lightning.

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train(config, n_classes=50):

    # We recover the original path of the dataset:
    path = Path('data/ESC-50-master')

    # Load data
    train_data = ESC50Dataset(path=path, folds=[1, 2, 3])
    val_data = ESC50Dataset(path=path, folds=[4])
    test_data = ESC50Dataset(path=path, folds=[5])

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["batch_size"])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"])

    pl.seed_everything(0)

    # Initialize the network
    model = AudioNet(config["base_filters"], n_classes, config["lr"])
    trainer = pl.Trainer(max_epochs=1, gpus=0, progress_bar_refresh_rate=0,
                         callbacks=[TuneReportCallback({"val_acc": "val_acc"}, on="validation_end")])
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":

    config = {"lr": tune.loguniform(1e-4, 1e-1),
              "batch_size": tune.choice([8, 16]),
              "base_filters": tune.choice([16, 32])}

    scheduler = ASHAScheduler(
        max_t=1,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", "base_filters"],
        metric_columns=["val_acc"])

    analysis = tune.run(
        tune.with_parameters(train, n_classes=50),
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        metric="val_acc",
        mode="max",
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune")

    print("Best hyperparameters found were: ", analysis.best_config)
