#!/usr/bin/env python
# coding: utf-8

import torch, torchaudio
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


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
    
    def __init__(self, n_classes = 50, base_filters = 32, lr=3.e-4, optimizer_name="Adam"):
        super().__init__()
        
        # Model
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

        #Hyperparameters managed by Optuna
        self.lr = lr
        self.optimizer_name=optimizer_name
        
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
        #Obtain hyperparams combination
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def train_with_optuna_support(trial: optuna.trial.Trial):
    # This is the main training function requested by the exercise.
    # We use folds 1,2,3 for training, 4 for validation, 5 for testing.
    
    # Load data
    train_data = ESC50Dataset(folds=[1,2,3])
    val_data = ESC50Dataset(folds=[4])
    test_data = ESC50Dataset(folds=[5])

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)

    pl.seed_everything(0)

    # Exploit Optuna support for testing different ls and optimizer type.
    # Learning rate on a logarithmic scale.
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    # Optimizer to be used.
    optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD", "Adam"])


    # Initialize the network
    audionet = AudioNet(lr=lr, optimizer_name=optimizer_name)
    trainer = pl.Trainer(logger=True,
        max_epochs=5,
        gpus=-1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_acc')])

    hyperparameters = dict(lr=lr, optimizer_name=optimizer_name)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(audionet, train_loader, val_loader)

    return trainer.callback_metrics['val_acc'].item()

if __name__ == "__main__":

    # Use MedianPruner to early stop unpromising trials
    # according to median stopping rule. 
    # In this example, the pruning starts after the first trial.
    # Besides, to make the experiment reproducible, we fix the random seed
    sampler = TPESampler(seed=10)
    study = optuna.create_study(sampler=sampler, direction="maximize", pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=0))
    study.optimize(train_with_optuna_support, n_trials=10)

    print("Total number of trials: ", len(study.trials))

    btrial = study.best_trial
    print("Best score: ", btrial.value)

    print("  Params: ")
    for key, value in btrial.params.items():
        print(key, ": ", value)


