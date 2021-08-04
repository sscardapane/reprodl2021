# Reproducible Deep Learning
## Extra: Auto-PyTorch for neural architecture search

### Author: [Lorenzo Giusti](https://github.com/lrnzgiusti)

[[Official reprodl website](https://www.sscardapane.it/teaching/reproducibledl/)]
[AutoML Groups Freiburg and Hannover](http://www.automl.org/)
[[Official Auto-PyTorch Repository](https://github.com/automl/Auto-PyTorch/)]


> ⚠️ extra branches implement additional exercises created by the students of the 
> course to explore additional libraries and functionalities. They can be read 
> independently from the main branches. Refer to the original authors for more information.


## Installation

Clone repository

```sh
$ cd install/path
$ git clone https://github.com/automl/Auto-PyTorch.git
$ cd Auto-PyTorch
```
If you want to contribute to this repository switch to our current development branch

```sh
$ git checkout development
```

Install pytorch: 
https://pytorch.org/

Install Auto-PyTorch:

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ python setup.py install
```



## Instructions

We want to use ``Auto-PyTorch`` as a framework to optimize traditional ML pipelines and their hyperparameters. The following steps present a clear way to initialize the framework adopted and run the relative experiments.


### Step 0: Import AutoPyTorch module for a learning task

For simplicity of the experiments, we've chosen a classification task.

```py
from autoPyTorch import AutoNetClassification
```


### Step 1: Load Data

We use the [ESC50](https://github.com/karolpiczak/ESC-50) dataset. The dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification and consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories.

For a more convienient way to load the data, we've overloaded the PyTorch's *Dataset* class with a custom one for this specific dataset.

```py
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

```

##### Step 1.1: Divide the dataset into train, validation and test sets

We use folds 1,2,3 for training, 4 for validation, 5 for testing.

```py
train_data = ESC50Dataset(folds=[1,2,3])
val_data = ESC50Dataset(folds=[4])
test_data = ESC50Dataset(folds=[5])

X_train, y_train = train_data[0]
y_train = [y_train]

X_val, y_val = val_data[0]
y_val = [y_val]
```


##### Step 1.2: Concatenate the data

This step is done to bulk the batches into a single tensor.

```py
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

```

### Step 2: Initialize the architecture search object

For a time-based budged use the following code:

```py
autoPyTorch = AutoNetClassification(config_preset="tiny_cs",   
                                    log_level='debug',
                                    budget_type='time',
                                    num_iterations=5, 
                                    max_runtime=950,
                                    min_budget=50,
                                    max_budget=500)
```

For an epoch-based budget you simply have to change the respective parameter parameter:

```py
autoPyTorch = AutoNetClassification(config_preset="tiny_cs", 
                                    log_level='debug',
                                    budget_type='epochs',  # <---- 
                                    num_iterations=5,
                                    max_runtime=950,
                                    min_budget=50, 
                                    max_budget=500)
```



### Step 3: Fit the neural network

For a vanilla training you can just pass the training samples and the ratio to split the samples into training & validation sets

```py
results_fit = autoPyTorch.fit(X_train, y_train, validation_split=0.3)
```

For a more sophisticated training, you can add many options, without loss of generality you can use the following:

```py
results_fit = autoPyTorch.fit(optimize_metric='accuracy',
                          cross_validator='k_fold',
                          early_stopping_patience=3,
                          loss_modules=['cross_entropy', 'cross_entropy_weighted'],
                          log_level="debug",
                          X_train=X_train,
                          Y_train=Y_train,
                          validation_split=0.3,
                          cuda=True)
```





### Step 4: Measure the performance obtained


```py
import sklearn.metrics

y_pred = autoPyTorch.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
```


### Step 5: Save results into a JSON file


```py
import json
with open("reprodl/logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)
```

