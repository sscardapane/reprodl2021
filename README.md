# Reproducible Deep Learning
## Extra: Optuna for hyperparameters fine-tuning
### Author: Gabriele D'Acunto ([OfficiallyDAC](https://github.com/OfficiallyDAC))
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)]

## Goals

- [ ] Add [Optuna](https://optuna.readthedocs.io/en/latest/installation.html) support for hyperparameters fine-tuning in combination with PyTorch Lightning.
- [ ] Run Optuna along with [Hydra](https://hydra.cc/) by using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper/) plugin (it requires `hydra-core>=1.1.0`, currently only pre-release version is available).

## Prerequisites

1. Uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. Install [Optuna](https://optuna.readthedocs.io/en/latest/installation.html) by using pip (recommended):

```bash 
pip install optuna
```

## Instructions
### Task 1: run Optuna along with PyTorch Lightning
In the first experiment we try to combine `optuna` with `pytorch_lightning` module in order to validate both the learning rate (`lr`) of the model and the optimizer (`optimizer_name`).
More precisely, we let vary the `lr` between [1.e-4, 1.e-3]. Furthermore, we test two different optimizers (`SGD`, `Adam`). The code is implemented in a Python script called `train.py`.

1. Convert `"Initial Notebook.ipynb"` to a Python script by running:
```bash
jupyter nbconvert --TemplateExporter.exclude\_markdown=True --TemplateExporter.exclude\_input_prompt=True --to script --output "train" "Initial Notebook.ipynb"
```
Then, reorganize the notebook similarly to _exercise1\_git_. 

2. Import `optuna` library at the beginning of the code, including the pruning module and a sampler as follows:
```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPrune
from optuna.samplers import TPESampler
```
3. Build a training function equipped with `optuna` support named `train_with_optuna_support()`. The backbone of this function is essentially constituted by 3 major blocks:

  1. Load and wrap data using dataloaders:
 
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
4. Print the tuning procedure results while running the code. To make this experiment reproducible, we fixed the `sampler` seed.

### Task 2: call Optuna from Hydra by using Optuna Sweeper Plugin
In this experiment we equip Hydra with the aformentioned plugin and we exploit once again Optuna for parameters optimization. The code is implemented in a Python script termed `train_HydraPlusOptuna.py`.

1. In order to be able to install the plugin, we need `hydra-core>=1.1.0` (pre-release version is available). Therefore, please run:
```bash 
pip install --pre hydra-core
pip install hydra-optuna-sweeper --upgrade
```
2. Add some logging and enable `hydra` [colorlog](https://hydra.cc/docs/plugins/colorlog/) plugin (optional).
Essentially, we follow the same steps as in _exercise2_hydra_. Put at the top of the script the following:
```python
import logging
logger = logging.getLogger(__name__)
```
Thus, install `colorlog` plugin:
```bash
pip install hydra_colorlog
```

3. Make a folder named `config` and create inside it a config file named `dafault.yaml`.
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
Please notice that it is still not possible to specify any `optuna` pruning method.

4. Launch the script by running:

```bash
python train_HydraPlusOptuna.py --multirun
```