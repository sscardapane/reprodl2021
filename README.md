# Reproducible Deep Learning
## Extra: Optuna for hyperparameters fine-tuning
### Author: [OfficiallyDAC](https://github.com/OfficiallyDAC)
[[Official reprodl website](https://www.sscardapane.it/teaching/reproducibledl/)]

## Goals

- [ ] Add [Optuna](https://optuna.readthedocs.io/en/latest/installation.html) support for hyperparameters fine-tuning in combination with PyTorch Lightning.
- [ ] Run Optuna along with [Hydra](https://hydra.cc/) by using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper/) plugin (it requires `hydra-core>=1.1.0`, currently only pre-release version is available).

## Prerequisites

1. Complete [Exercise 1](https://github.com/sscardapane/reprodl2021/tree/exercise1_git).
2. Install [Optuna](https://optuna.readthedocs.io/en/latest/installation.html) by using pip (recommended):

```bash 
pip install optuna
```

## Instructions
### Task 1: run Optuna along with PyTorch Lightning
In the first experiment we try to combine `optuna` with `pytorch_lightning` module in order to validate some hyperparameters of the model. The purpose of this section is not to find optimal values for model hyperparameters, but rather to show how it is possible to exploit open-source tools to automate the search for hyperparameters. More precisely, in this experiment, we will implement a code able to tune both the learning rate (`lr`) and which optimizer (`optimizer_name`) to be used in an automatic fashion.
As an example, we let vary the `lr` between [1.e-4, 1.e-3]. Furthermore, we test two different optimizers (`SGD`, `Adam`). The code is implemented in a Python script called `train.py`.

1. Import `optuna` library at the beginning of the code, including the pruning module and a sampler as follows:
```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPrune
from optuna.samplers import TPESampler
```
2. Build a training function equipped with `optuna` support named `train_with_optuna_support()`. The backbone of this function is essentially constituted by 3 major blocks:
    1. Load and wrap data using dataloaders
    2. Introduce `optuna` support:
    ```python
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD", "Adam"])
    ```
    3. Initialize and fit the model using `pytorch_lightning`. During the training step, we also exploit `optuna.pruners.MedianPruner()` method, which stops unpromising trials according to the median early stopping rule.
    ```python
    trainer = pl.Trainer(logger=True,
          max_epochs=25,
          gpus=-1 if torch.cuda.is_available() else None,
          callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_acc')])
    ```
3. Print the tuning procedure results while running the code. To make this experiment reproducible, we fixed the `sampler` seed.

### Task 2: call Optuna from Hydra by using Optuna Sweeper Plugin
In this experiment we equip Hydra with the aformentioned plugin and we exploit once again Optuna for parameters optimisation. In particular, Hydra is an open-source tool which provides an efficient way to manage hyperparameters configuration both from `yaml` file and command line. Therefore, it is worth to experiment with Hydra Optuna Sweeper in order to combine two useful tools for hyperparameters managing and optimisation. The code is implemented in a Python script termed `train_HydraPlusOptuna.py`, whereas the configuration file `default.yaml` is located in the folder `config`.

1. Create a copy of `train.py` and rename it `train_HydraPlusOptuna.py`. Thus, apply the modification below on the latter.

2. In order to be able to install the plugin, we need `hydra-core>=1.1.0` (currently only pre-release version is available). Therefore, please run:
```bash 
pip install --pre hydra-core
pip install hydra-optuna-sweeper --upgrade
```
3. Add some logging and enable `hydra` [colorlog](https://hydra.cc/docs/plugins/colorlog/) plugin (optional).
Essentially, we follow the same steps as in _exercise2_hydra_. Put at the top of the script the following:
```python
import logging
logger = logging.getLogger(__name__)
```
Thus, install `colorlog` plugin:

```bash
pip install hydra_colorlog
```

4. Make a folder named `config` and create within it a configuration file named `dafault.yaml`.
First of all, override `hydra` defaults:
```yaml
defaults:
  - override hydra/job_logging: colorlog #optional
  - override hydra/hydra_logging: colorlog #optional
  - override hydra/sweeper: optuna
  - override hydra/sweeper/sampler: tpe
```
Then, we proceed by including all other `optuna` parameters used in __Task 1__:

```yaml
hydra:
  sweeper:
    sampler:
      seed: 10
    direction: maximize
    study_name: train_with_optuna_support
    storage: null
    n_trials: 10
    n_jobs: 1
    
    search_space:
      lr:
        type: float
        low: 1e-4
        high: 1e-3
        log: True

      optimizer_name:
        type: categorical
        choices: ['SGD', 'Adam']

lr: 3.e-4
optimizer_name: 'Adam'
```
Please notice that, unfortunately, it is still not possible to specify any `optuna` pruning method.

5. Modify the python code in order load the configuration file above by using `hydra`. Essentially, you need to decorate `train_with_optuna_support()` function to accept the configuration file and set all hyperparameters in the script through `cfg` object.

6. Launch the script by running:

```bash
python train_HydraPlusOptuna.py --multirun
```

7. Last but not least, the plugin allows us to override `optuna` search space parametrization directly from command line. For instance, below we modify the lower bound of `lr` search space range:

```bash
python train_HydraPlusOptuna.py --multirun 'lr=tag(log, interval(1.e-5,1.e-3))'
```
Please notice that in the command above, the `log` tag allows us to use a `LogUniformDistribution` rather than a `UniformDistribution` (as it happens when using only `interval`). For further details on the search space override, please refer to the plugin [main page](https://hydra.cc/docs/next/plugins/optuna_sweeper) 
