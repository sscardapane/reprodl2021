#!/usr/bin/env python
# coding: utf-8

import torch, torchaudio
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

import os
import pandas as pd
import talos


class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50

    def __init__(self, path='data/ESC-50', sample_rate: int = 8000, folds=[1]):
        # Load CSV & initialize all torchaudio.transforms:
        # Resample --> MelSpectrogram --> AmplitudeToDB
        self.path = path
        self.csv = pd.read_csv(os.path.join(path, 'meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        self.resample = torchaudio.transforms.Resample( orig_freq=44100, new_freq=sample_rate)
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(os.path.join(self.path, 'audio', row['filename']))
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


class AudioNet(nn.Module, talos.utils.TorchHistory):

    def __init__(self, n_classes=50, base_filters=32):
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


def optimize(x_train, y_train, x_val, y_val, params):
    # This is the main training function requested by the exercise.
    # We use folds 1,2,3 for training, 4 for validation, 5 for testing.

    # Initialize the network
    net = AudioNet()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=params['lr'])

    # Initialize history of net
    net.init_history()

    for epoch in range(params["epochs"]):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(x_train)

        # calculate accuracy
        prediction = torch.max(outputs, 1)[1]
        metric = f1_score(y_train.data, prediction.data)

        # calculate loss + backward + optimize
        loss = loss_func(outputs, y_train)
        loss.backward()
        optimizer.step()

        # calculate accuracy for validation data
        output_val = net(x_val)
        prediction = torch.max(output_val, 1)[1]
        val_metric = f1_score(y_val.data, prediction.data)

        # calculate loss for validation data
        val_loss = loss_func(output_val, y_val)

        # append history
        net.append_loss(loss.item())
        net.append_metric(metric)
        net.append_val_loss(val_loss.item())
        net.append_val_metric(val_metric)

    # Get history object
    return net, net.parameters()


if __name__ == "__main__":

    # Load data
    train_data = ESC50Dataset(folds=[1, 2, 3])
    val_data = ESC50Dataset(folds=[4])
    test_data = ESC50Dataset(folds=[5])

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    # Extract Train and Val set
    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(train_loader))

    params = {"lr": [1e-3, 1e-4, 1e-5, 1e-6],
              "batchsize": [2, 4, 8, 16],
              "epochs": [10, 20, 30, 40]}

    scan_object = talos.Scan(x=x_train,
                             y=y_train,
                             x_val=x_val,
                             y_val=y_val,
                             params=params,
                             model=optimize,
                             experiment_name='talos')
