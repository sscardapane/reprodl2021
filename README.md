# Reproducible Deep Learning
## Extra: Talos: Hyperparameter Optimization for Keras, TensorFlow and PyTorch
### Author: [Valerio Guarrasi](https://github.com/guarrasi1995), [Andrea Marcocchia](https://github.com/andremarco), [Eleonora Grassucci](https://github.com/eleGAN23)

Before going into this branch, please look at the main branch in order to understand the project details.
> :warning: **extra** branches implement additional exercises created by the students of the course to explore additional libraries and functionalities. They can be read independently from the main branches. Refer to the original authors for more information.

&nbsp;

## Prerequisites

1. Uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. Install generic requirements:

```bash
pip install -r requirments.txt
```
3. Install the *branch specific* requirments:
```bash
pip install talos
```
In order to run the experiment explained in this branch, we used the `1.0` *Talos* version. In order to guarantee reproducibility, it is possible to install the specific version:

```bash
pip install talos==1.0
```
&nbsp;

## Goal

The aim of this exercise is to understand how to set talos in a simple way, in order to remain in complete control of the PyTorch models, without bothering with mindless parameter hopping and confusing optimization solutions that add complexity instead of reducing it. Talos incorporates grid, random, and probabilistic hyperparameter optimization strategies, with focus on maximizing the flexibility, efficiency, and result of random strategy.
&nbsp;


## Instructions
Finding the right hyperparameters for your deep learning model can be a tedious process. With the right process in place, it will not be difficult to find state-of-the-art hyperparameter configuration for a given prediction task.

In this branch we implement a hyperparameter scanning based on an already created model. In addition to the input model, a hyperparameter scan with *Talos*, via the `talos.Scan()` command and a parameter dictionary is added.

To get started with our experiment we need to have three things:
1. Prepare the input model;
2. Define the parameter space;
3. Configure the experiment.

First of all, we have to define the hyper-parameters dictionary:

```json
params = {"lr": [1e-3, 1e-4, 1e-5, 1e-6],
          "batchsize": [2, 4, 8, 16], 
          "epochs": [10, 20, 30, 40]
         }
```
Depending on the needs, many different parameters could be defined. If different losses, optimizers, and activations functions are defined in the dictionary that we want to include in the scan, we should need to import those functions/classes direct from the main module (PyTorch, Keras, ...).

Once the model and parameters are ready, we can start the experiment with the following command:

```python
scan_object = talos.Scan(x=x_train,
                         y=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         params=params,
                         model=optimize,
                         experiment_name='talos')
```

It is possible to analyze the results of a scanning process using the *analyze* function:
```python
talos.Analyze(scan_object)
```


For further details on the search space possible settings, please refer to the plugin [page](https://autonomio.github.io/talos/#/Optimization_Strategies?id=optimization-strategies).

Congrats! The extra exercise is concluded.
