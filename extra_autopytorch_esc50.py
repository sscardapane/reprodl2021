# -*- coding: utf-8 -*-

from autoPyTorch import (AutoNetClassification,
                         AutoNetMultilabel,
                         AutoNetRegression,
                         AutoNetImageClassification,
                         AutoNetImageClassificationMultipleDatasets)

import torch, torchaudio
from torch import nn
from torch.nn import functional as F

#import pytorch_lightning as pl
#from pytorch_lightning.metrics import functional

from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os as os
import openml
import json

print("Using GPU:", torch.cuda.is_available())

# Substitute this with your actual path. This is the root folder of ESC-50, where
# you can find the subfolders 'audio' and 'meta'.
path = 'reprodl/data/ESC-50-master'
datapath = Path(path)
print("Path Exist:", datapath.exists())
# Using Path is fundamental to have reproducible code across different operating systems.

# next lines are kept only for see if the data is loaded correctly
csv = pd.read_csv(datapath / Path('meta/esc50.csv'))
print(csv.head())
x, sr = torchaudio.load(datapath / 'audio' / csv.iloc[0, 0], normalize=True)
plt.plot(x[0, ::5])
# Useful transformation to resample the original file.
torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)(x).shape
# Another useful transformation to build a Mel spectrogram (image-like), so that
# we can apply any CNN on top of it.
h = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(x)
# Convert to DB magnitude, useful for scaling.
# Note: values could be further normalize to significantly speed-up and simplify training.
h = torchaudio.transforms.AmplitudeToDB()(h)
plt.imshow(h[0])

class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self, path: Path = Path(path), 
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

# We use folds 1,2,3 for training, 4 for validation, 5 for testing.
train_data = ESC50Dataset(folds=[1,2,3])
val_data = ESC50Dataset(folds=[4])
test_data = ESC50Dataset(folds=[5])

X_train, y_train = train_data[0]
y_train = [y_train]

X_val, y_val = val_data[0]
y_val = [y_val]


for i in range(1, 50):
  xb, yb = train_data[i]
  X_train = torch.cat((X_train, xb), 0)
  y_train.append(yb)

for i in range(1, 20):
  xb, yb = val_data[i]
  X_val = torch.cat((X_val, xb), 0)
  y_val.append(yb)

y_train = np.array(y_train)
y_val = np.array(y_val)

autonet = AutoNetClassification(config_preset="tiny_cs", budget_type='epochs', min_budget=50, max_budget=1000, num_iterations=5, log_level='debug')
# Get the current configuration as dict
current_configuration = autonet.get_current_autonet_config()

# Get the ConfigSpace object with all hyperparameters, conditions, default values and default ranges
hyperparameter_search_space = autonet.get_hyperparameter_search_space()

# Fit (might need larger budgets)
results_fit = autonet.fit(optimize_metric='auc_metric',
                      cross_validator='k_fold',
                      early_stopping_patience=3,
                      loss_modules=['cross_entropy', 'cross_entropy_weighted'],
                      log_level="debug",
                      X_train=X_train,
                      Y_train=y_train,
                      X_valid=X_val,
                      Y_valid=y_val,
                      cuda=True
                      )

# Save fit results as json
with open("reprodl/logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)

