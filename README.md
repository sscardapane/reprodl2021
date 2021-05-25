# Reproducible Deep Learning
## Extra: DVC for experiments management

### Author: [FedericoCinus](https://github.com/FedericoCinus)

[[Official reprodl website](https://www.sscardapane.it/teaching/reproducibledl/)]

> ⚠️ extra branches implement additional exercises created by the students of the 
> course to explore additional libraries and functionalities. They can be read 
> independently from the main branches. Refer to the original authors for more information.

## Goals

- [ ] Initialize ``dvc`` support for experiment management.
- [ ] Prepare parameters and metrics configuration files, and checkpoints.
- [ ] Run experiments and access results.


## Prerequisites

1. Uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. If this is your first exercise, run *train.py* to check that everything is working correctly.
3. Install [dvc](https://dvc.org).
4. Install [dvclive](https://pypi.org/project/dvclive/).
5. Install [yaml](https://pypi.org/project/PyYAML/).

```bash
pip install <package_name>
```


## Instructions

The aim of this tutorial is to use ``dvc`` as a tool to define data pipelines in ML, for experiments management purposes. The following steps present a clear way to initialize the ``dvc`` support for experiment management, prepare configuration files and run experiments.


### Step 0: create simulation folder.

In the first step we create an experiment-data folder, which is going to contain all experiments results.
```bash
mkdir experiments-data
```
Details:
1 - We aim to access all experiments in a unique dvc version; thus, we create a unique experiment-data folder. (This type of structuring is not unique, see [Organization Patterns](https://dvc.org/doc/user-guide/experiment-management) for more details).
2 - We want to exploit the dvc functionalities and let it track all the generated data; therefore we insert the experiments folder in the ``.gitignore`` file.

### Step 1: configure parameters.
Secondly, we create a ``.yaml`` file called ``params.yaml``.
```bash
vi params.yaml
```
This file contains the definition of the parameters.
```python
train:
  data_path: "data/ESC-50"
  base_filters: 32
  n_classes: 50
  batch_size: 8
  gpus: 1
  max_epochs: 1
  seed: 42
```
Details:
1 - Each step of the ML pipeline is called ``stage``. Here we consider a unique training stage: ``train``. 
2 - All the parameters refer to this stage.

To read the parameters in the training function, we insert the following lines in ``train.py`` and modify the ``train`` function to accept the parameters:
```python
params = yaml.safe_load(open('params.yaml'))['train']
train(params)
```

### Step 2: initialize dvc pipeline.
In this step we aim to initialize the pipeline. ``dvc`` needs the following 3 details: name of the stage, command, and dependencies. There are two ways to initialize a pipeline but both rely on the same information stated above:
##### 1 - create a dvc.yaml file:
```python
stages:
  train:
    cmd: python train.py
    deps:
      - train.py
``` 
Details: 1. "train" is the name of the stage, "cmd" is line command, "deps" are the dependencies.

##### 2 - run ``dvc run``:
We now initialize the simulation pipeline creating the so-called stages:
```bash
dvc run -n train \
        -d train.py \
        python train.py 
```          
Details: 1. `-n` is used to define the name of the stage, 2. `-d` the dependencies. Other parameters can be used to specify the output (`-o`) or the parameters (`-p`); 3. if some files are missing ``dvc`` creates the missing files.

The folders structure should be like this:
```bash
.
├── experiments-data
+├── dvc.yaml
+├── dvc.lock
├── experiments-data.dvc
├── params.yaml
├── train.py
└── src
   ├── ...
```

To display the pipeline we can type:
```bash
  dvc dag
```

### Step 3: configure metrics.
In the third step we are going to track some metrics, by: 1. modifying the ``dvc.yaml`` file, 2. saving a metrics file.

1. We add the "metrics" details in the "train" stage, specifying the "json" file containing the metrics and if we want to store these values in cache:
    ```python
    stages:
      train:
        cmd: python train.py
        deps:
          - train.py
        metrics:
        - summary.json:
            cache: true
    ``` 
    The summary.json should be like this: i) "stages" key; ii) the specified stage name ("train"); iii) the name of the metric that we choose ("accuracy").
    ```python
    {'stages': {'train': {'accuracy': 3.5847707}}}
    ``` 
3. We add few lines to save the chosen metric in the json file: i) we use the `Trainer` instance from PyTorch Lightning to access the "logged metrics" dictionary and extract the training loss; ii) we extract the scalar value from a 0-dim tensor saved in the GPU; iii) we save in the summary.json the metric value.
    ```python
      accuracy = trainer.logged_metrics['train_loss'].data.cpu().numpy().reshape(1)[0]
    
      summary_data = {'stages':{'train':{'accuracy': accuracy}}}
      with open('summary.json', 'w') as curr_file:
          curr_file.write(str(summary_data))
      ``` 
  N.B. in this last step, the first line of code is specifically designed for the PyTorch example used.


### Step 5: insert checkpoints.

In this step we record some checkpoints in the experiments using ``dvclive``.
Firstly, we import the package:
    ```python
      import dvclive
      ``` 
Secondly, we log the metric and make the checkpoint with the "next_step" method:
    ```python
      dvclive.log('train_loss', train_loss)
      dvclive.next_step()
      ``` 
Details:
1. dvc creates the logs.json, logs.html files and the logs folder, in order to be able to access the logged metrics.
2. In the logs.html, the plot of the chosen metric is displayed.


### Step 6: do experiments.
To run the experiments we can use two commands.
1. ```bash
      dvc exp run -n first_train
   ``` 
2. ```bash
     dvc repro
   ``` 
Details: 1. the `-n` parameter is used to specify a name for the experiment. 2. The command "repro" is an old command for reproducibility and, since it uses the same yaml file which describes the pipeline, it produces the same output. 3. We can access the experiments and confront them by using:

```bash
     dvc exp show
```
In this case, a table with the named experiments, parameters and result metrics are displayed.


### Step 7: commit results.
Finally, the following steps are required to commit the experiments files:

1. we add the experiments-data folder in dvc:
    ```bash
        dvc add experiments-data
    ```
1. we add the dvc file which tracks the experiments-data folder in git:
    ```bash
        git add experiments-data.dvc
    ```
1. we commit the changes in the stage:
    ```bash
        git commit -m "Committing the first experiment."
    ```
