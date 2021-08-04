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

*Talos* is an hyperparameter optimization tool. The main objective of *Talos* is to simplify the hyperparameter optimization in a machine learning model (eg. PyTorch, Tensorflow, Keras, ...) in order to improve the experiments setup. At the same time, with *Talos* you keep the control of your model, without the classical difficulties of AutoML tools. Moreover, *Talos* does not introduce any new syntax and boilerplate code.

*Talos* can work with:

* grid search (cartesian);
* random grid search;
* probabilistic optimization.

## Instructions
In this branch we implement a hyperparameter scanning based on the already created model. The hyperparameter scan with *Talos* is performed via the `talos.Scan()` command, that requires a parameter dictionary.

To get started with our experiment we have to set up three things:
1. Prepare the input model;
2. Define the parameter space;
3. Configure the experiment.

First of all, we have to define the hyperparameters dictionary:

```json
params = {"lr": [1e-3, 1e-4, 1e-5, 1e-6],
          "batchsize": [2, 4, 8, 16], 
          "epochs": [10, 20, 30, 40]
         }
```
Depending on the task, different parameters could be defined. If different losses, optimizers, and activations functions are defined in the dictionary that we want to include in the scan, we should need to import those functions/classes direct from the main module (PyTorch, Keras, ...).

Once the model and parameters are ready, we can start the hyperparameter scanning with the following command:

```python
scan_object = talos.Scan(x=x_train,
                         y=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         params=params,
                         model=optimize,
                         experiment_name='talos')
```
Once the experiment is done, it is possible to take a look to the results of the scanning process.

Looking at the `scan_object.details` attribute, we can obtain the meta-information of the experiment. We can access the epoch entropy values for each hyperparameter round using `scan_object.learning_entropy`

Using the *analyze* function some more insights could be displayed:

```python
analysis = talos.Analyze(scan_object)
```
 
The `Analyze` has several attributes to explore the optimization strategy and results:

* `analysis.data` returns the results dataframe;
* `analysis.best_params` returns the best hyperparameters;
* `plot_*` gives a visual insight of different aspects such as an histogram for the selected metric or a correlation heatmap where the metric is plotted against hyperparameters.

In the end, through *Talos* we can easily access the best saved model and weights for each hyperparameter permutation just running the following commands:
```python
# Retrieve models
scan_object.saved_models

# Retrieve weights
scan_object.saved_weights
```

When the best setting and hyperparameter values have been found, *Talos* gives the possibility to create a deployment package with `Deploy()`.

For further details on the search space possible settings, please refer to the plugin [page](https://autonomio.github.io/talos/#/Optimization_Strategies?id=optimization-strategies).

Congrats! The extra exercise is concluded.
