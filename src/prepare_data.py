import torch, torchaudio
from torch import nn

from pathlib import Path
import pandas as pd

import sys
import yaml
import pickle
import os


class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50

    def __init__(self, path: Path = Path('data/initial/ESC-50'),
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare_data.py"+\
                         " path-to-data mode\n")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    mode = sys.argv[2]
    obj_name = mode + ".pickle"
    output_destination = os.path.join("data","prepared", obj_name)
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["prepared"]
    sample_rate = params["sample_rate"]
    folds = params["folds_"+mode]

    object_data = ESC50Dataset(data_path, sample_rate, folds)

    with open(output_destination, 'wb') as out_dest:
        pickle.dump(object_data, out_dest)
