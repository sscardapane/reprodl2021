import torch, torchaudio
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

import torchmetrics

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import RegressionPerformanceTab


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

        # self.csv.columns
        
        
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

        self.results = tuple()
        self.training_results = tuple()
        self.validation_results = tuple()
        
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

        str_acc = torchmetrics.functional.accuracy(y_hat, y)

        self.training_results = str_acc, y_hat, y

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        val_acc = torchmetrics.functional.accuracy(y_hat, y)

        self.validation_results = val_acc, y_hat, y, x

        return acc
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        #logits = self(x)
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        #loss = F.gaussian_nll_loss(logits, y)
        #self.log("test_loss", loss)

        self.results = acc, y_hat, y, x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_training_results(self):
        return self.training_results

    def get_validation_results(self):
        return self.validation_results

    def get_results(self):
        return self.results


def train():
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

    # Initialize the network
    audionet = AudioNet()
    trainer = pl.Trainer(gpus=0, max_epochs=2)
    trainer.fit(audionet, train_loader, val_loader)
    
    # Perform testing phase
    result = trainer.test(audionet, test_loader)

    # Get results obtained for different phases
    (str_accuratezza, str_y_hat, str_y) = audionet.get_training_results()
    (val_accuratezza, val_y_hat, val_y, val_x) = audionet.get_validation_results()
    (accuratezza, y_hat, y, x) = audionet.get_results()

    return accuratezza, y_hat, y, x, str_accuratezza, str_y_hat, str_y, val_accuratezza, val_y_hat, val_y, val_x, list_columns_header


if __name__ == "__main__":
    acc, y_hat, y, x, str_acc, str_y_hat, str_y, val_acc, val_y_hat, val_y, val_x, list_columns_header = train()

    # Dictionary preparation for both reference and production values
    reference = {"target": val_y.tolist(), "prediction": val_y_hat.tolist()}
    production = {"target": y.tolist(), "prediction": y_hat.tolist()}

    # Pandas' DataFrames preparation
    df_reference = pd.DataFrame(reference)
    df_production = pd.DataFrame(production)

    print("Pandas DataFrames prepared!")

    # Dashboard instatiation with a custom set of tabs provided as list
    audionet_report = Dashboard(tabs=[RegressionPerformanceTab()])

    # Invoking calculate method to process results and prepare the dashboard
    audionet_report.calculate(df_reference, df_production, column_mapping = None)

    # With the show() method we diplay the dashboard inside the Notebook
    audionet_report.show()

    # Using the save() method, instead, we can save the report to an HTML file
    # (highly suggested for particularly big datasets)
    audionet_report.save("reports/audionet_report.html")

