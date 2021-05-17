# Reproducible Deep Learning
## Extra: Optuna for hyperparameters fine-tuning
### Author: Gabriele D'Acunto
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)]

## Goals

- [ ] Adding Optuna support for hyperparameters fine-tuning in combination with PyTorch Lightning.

## Prerequisites

1. Uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. Install [Optuna](https://optuna.readthedocs.io/en/latest/installation.html):

```bash (recommended)
pip install optuna
```

## Steps
In this experiment we try to combine Optuna with PyTorch Lightning module in order to validate both the learning rate (**lr**) of the model and the optimizer (**optimizer_name**).
More precisely, we let vary the **lr** within [1.e-4, 1.e-3]. Furthermore, we test two different optimizers ('SGD', 'Adam')

1. Convert "Initial Notebook.ipynb" to a Python script by running:

``` jupyter nbconvert --TemplateExporter.exclude\_markdown=True --TemplateExporter.exclude\_input_prompt=True --to script --output "train" "Initial Notebook.ipynb" ```

and reorganize the notebook similarly to _exercise1\_git_. 

2. Import Optuna library at the beginning of the code, including the pruning module and a sampler as follows
```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPrune
from optuna.samplers import TPESampler
```
3. Build a training function equipped with Optuna support (**train_with_optuna_support**). The backbone of this function is essentially constituted by 3 major blocks:

  1. Load data and wrap using dataloaders:
 
  2. Introduce Optuna support:
  ```python
  lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
  optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD", "Adam"])
  ```
  3. Initialize and fit the model using PyTorch Lightning. During the training step, we also exploit Optuna MedianPruner method, which stops unpromising trials according to the median rule.
  ```python
  trainer = pl.Trainer(logger=True,
        max_epochs=5,
        gpus=-1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_acc')])
  ```
4. Print tuning procedure results while running the code. To make this experiment reproducible, we fixed the sampler seed.

