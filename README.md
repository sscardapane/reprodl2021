# Reproducible Deep Learning
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)] [[Slides](https://docs.google.com/presentation/d/1_AYIcCyVI59QiiXqU4Sn7VzwtVyfqv-lG36EPFzeSdY/edit?usp=sharing)]
## Extra: experiment_management_dvc
### Author: [Riccardo Denni](https://github.com/rdenni)
[DVC experiments management](https://dvc.org/doc/start/experiments)


Before going into this branch, please look at the main branch in order to understand the project details.
>:warning: extra branches implement additional exercises created by the students of the course to explore additional libraries and functionalities.
> They can be read independently from the main branches. Refer to the original authors for more information.

&nbsp;

This is an extra branch of **exercise1_git_completed** whose intent is to show an introduction to how to manage experiments with dvc 

&nbsp;

## Goal

The goal is to show the first basic steps to run machine learning experiments with dvc.


## Prerequisites

1. Complete [Exercise 1](https://github.com/sscardapane/reprodl2021/tree/exercise1_git).
2. Install [dvc](https://dvc.org/doc/install) 
3. Install [yaml](https://pypi.org/project/PyYAML/)

## Instructions

### Dvc Initialization

First let's initialize dvc 

```bash
dvc init
```

A few internal files are created that should be added to Git:

```bash
git add .
```

```bash
git commit -m 'dvc initialized'
```

### Dvc Pipelines

A data pipelines is a series of processes that transforms data and produce a final result. It is composed of stages, a stage represent a process which form a step of the pipeline. Stages also connect code to its corresponding data input and output. 

Our idea is to keep the data transformation phase separate from the training, validation and testing phase, to do this we build a pipeline with several stages. 

#### Data Transformation Stages

Regarding the data transformation phase we create three stages: one is in charge of preparing the data for training, one for validation and one for testing.
In order to do this, we create two subdirectories inside the data directory, "initial" and "prepared". We move ESC-50 inside "initial".

Let's have dvc trace the "initial" directory

```bash
dvc add data/initial
```
Now we divide *train.py* into two programs: one deals with data transformation and the other with training, validation and testing. We create a *src* folder where we insert the codes. 

The program to transform the data is called *prepare_data.py*. It takes in input two command line arguments: the path from which to take the data and a variable *mode*, to be set equal to *train*, *val* or *test* which indicates, respectively, whether the data is to be prepared for training, validation or testing.

In preparing the data we observe that there are parameters such as the sample rate and the folders from which to take the data, so, inside the folder *reprodl2021* we create a file *params.yaml*

```python
prepared:
  sample_rate: 8000
  folds_train: [1, 2, 3]
  folds_val: [4]
  folds_test: [5]
``` 

The parameters *folds_train*, *folds_val* and *folds_test* have been created to avoid as much as possible that one or more folders are used for more than one operation between training, validation and testing.

Now we create another subdirectory of data that we call "prepared", where we will save the output of the program.

The program takes as input the path to the data and the mode variable, creates an ESC50Dataset object with the parameters specified in params.yaml and then saves the object in data/prepared

In order for it to run we must also import these modules in addition to the modules that were already being imported:

```python
import sys
import yaml
import pickle
import os
``` 

The ESC50Dataset class was left as it was, this is the code that was added:

```python
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare_data.py" +\
                         " path-to-data mode\n")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    mode = sys.argv[2]
    obj_name = mode + ".pickle"
    output_destination = os.path.join("data","prepared",\
                                       obj_name)
    os.makedirs(os.path.join("data", "prepared"),\
                 exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["prepared"]
    sample_rate = params["sample_rate"]
    folds = params["folds_"+mode]

    object_data = ESC50Dataset(data_path, sample_rate, folds)

    with open(output_destination, 'wb') as out_dest:
        pickle.dump(object_data, out_dest)
```

Now we are ready to create the three stages of the pipeline.

1. *prepare_train_data* stage

```bash
dvc run -n prepare_train_data\
 -p prepared.sample_rate -p prepared.folds_train\
 -d src/prepare_data.py -d data/initial\
 -o data/prepared/train.pickle\
  python3 src/prepare_data.py data/initial/ESC-50 train
```

2. *prepare_val_data* stage

```bash
dvc run -n prepare_val_data\
 -p prepared.sample_rate -p prepared.folds_val\
 -d src/prepare_data.py -d data/initial\
 -o data/prepared/val.pickle\
 python3 src/prepare_data.py data/initial/ESC-50 val

```

3. *prepare_test_data* stage

```bash
dvc run -n prepare_test_data\
 -p prepared.sample_rate -p prepared.folds_test\
 -d src/prepare_data.py -d data/initial\
 -o data/prepared/test.pickle\
 python3 src/prepare_data.py data/initial/ESC-50 test

```


Let's examine one at random among these dvc run commands, let's say the last one:
- ```-n prepare_test_data``` specify the name of the stage
- ```-p prepared.sample_rate -p prepared.folds_test``` defines the parameters of the stage
- ```-d src/prepare_data.py -d data/initial``` defines the dependencies of the stage, notice that there is the source code itself. A dependency is a file or a directory while a parameter is not. (https://dvc.org/doc/command-reference/params)
- ```-o data/prepared/test.pickle``` specifies the output of the program
- ```python3 src/prepare_data.py data/initial/ESC-50 test``` is the command to run in this stage, and it's saved to dvc.yaml


#### Data Train, Val & Test Pipeline

Now let's deal with the program that works with the prepared data, which is called *train.py*. It takes two inputs from the command line: one is the folder from which to take the ESC50Dataset objects and the other is the path where to save the metrics that the model produces in the testing phase. The ESC50Dataset objects it works with are *train.pickle*, *val.pickle* and *test.pickle*. Metrics will be saved in the  directory *reprodl2021/results*


Again, there are parameters such as the number of classes, the number of neural network filters, the batch size of the DataLoader, and the number of epochs. So we add these lines to the *params.yaml* file:

```python
audio_net:
  n_classes: 50
  base_filters: 16
data_loader:
  batch_size: 8
training:
  max_epochs: 4
``` 


In order for it to run we must also import these modules in addition to the modules that were already being imported:

```python
import json
import pickle
import sys
import os
import yaml
from prepare_data import ESC50Dataset
```

In the *AudioNet* class we add the *test_step* method

```python
def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = functional.accuracy(y_hat, y)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics
```

We also add to the program a new function called *test*

```python
def test(tr, test_d, b_s):
    #tr is trainer, test_d is test_data, b_s is batch_size
    test_loader = torch.utils.data.DataLoader(test_d,\
                                              batch_size=b_s)
    test_metrics = tr.test(dataloaders=test_loader)
    return test_metrics
```

Moreover we modify the function train in such a way that it returns the trained model, to do this we only need to add 
```python
return trainer 
```
at the end.

This code was also added to the program:

```python
if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    train_data_file = os.path.join(sys.argv[1], "train.pickle")
    val_data_file = os.path.join(sys.argv[1], "val.pickle")
    test_data_file = os.path.join(sys.argv[1], "test.pickle")

    os.makedirs(os.path.join("results"), exist_ok=True)
    test_scores_file = os.path.join(sys.argv[2],\
                                    "test_scores.json")


    with open(train_data_file, 'rb') as train_file:
        train_data = pickle.load(train_file)

    with open(val_data_file, 'rb') as val_file:
        val_data = pickle.load(val_file)

    with open(test_data_file, 'rb') as test_file:
        test_data = pickle.load(test_file)

    params = yaml.safe_load(open("params.yaml"))
    params_audio_net = params["audio_net"]
    params_data_loader = params["data_loader"]
    params_training = params["training"]

    n_classes = params_audio_net["n_classes"]
    base_filters = params_audio_net["base_filters"]

    batch_size = params_data_loader["batch_size"]
    max_epochs = params_training["max_epochs"]

    trainer = train(train_data, val_data, n_classes,\
                    base_filters, batch_size, max_epochs)
    test_metrics = test(trainer, test_data, batch_size)[0]

    with open(test_scores_file, 'w') as test_s_f:
        json.dump(test_metrics, test_s_f)
```

Now a json file will be saved with the loss and accuracy of the model in the testing phase.

We are ready to add the last stage to the pipeline:

```bash
dvc run -n train_and_test\
 -p audio_net.n_classes -p audio_net.base_filters\
 -p data_loader.batch_size -p training.max_epochs\
 -d src/train.py -d data/prepared/train.pickle\
 -d data/prepared/val.pickle -d data/prepared/test.pickle\
 -M results/test_scores.json\
 python3 src/train.py data/prepared results
```

```-M results/test_scores.json``` specifies the metric file produced by the stage and says that dvc does not track the metrics file.(if we wanted dvc to track the metrics file we should have put -m instead of -M). We don't have it tracked by dvc because we want to have it tracked by git, it's not a large file anyway. 

Now let's add the changes and do a git commit.

To reproduce the pipeline you just need this command:

```bash
dvc repro
```


Instead to see the pipeline structure

```bash
dvc dag
```

### Dvc Experiments

We are now ready to run the experiments with dvc. We queue them up and then run them one after the other. For each experiment there will be different values for the parameters base_filters and max_epochs.

Let's activate the reprodl environment

```bash
conda activate reprodl
```
Now, let's queue the experiments

```bash
dvc exp run --queue --set-param training.max_epochs=3\
 --set-param audio_net.base_filters=8

dvc exp run --queue --set-param training.max_epochs=3\
 --set-param audio_net.base_filters=16

dvc exp run --queue --set-param training.max_epochs=3\
 --set-param audio_net.base_filters=32

dvc exp run --queue --set-param training.max_epochs=4\
 --set-param audio_net.base_filters=8

dvc exp run --queue --set-param training.max_epochs=4\
 --set-param audio_net.base_filters=16

dvc exp run --queue --set-param training.max_epochs=4\
 --set-param audio_net.base_filters=32
```

To see the results of the experiments

```bash
dvc exp show --num 5 --no-timestamp --include-params audio_net.base_filters,training.max_epochs
```

- ```--num 5``` is to show the last 4 commits from HEAD.
- ```--no-timestamp``` is to not show the timestamps
- ```--include-params audio_net.base_filters,training.max_epochs``` is to show audio_net.base_filters and training.max_epochs in the table only

