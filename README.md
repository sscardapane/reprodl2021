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

We use the [digits](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) dataset already loaded into the [SkLearn](https://scikit-learn.org/stable/index.html) module.

```py
import sklearn.datasets
import sklearn.model_selection

X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=193)

```


### Step 2: Initialize the architecture search object

For a time-based budged use the following code:

```py

autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='debug',
                                    max_runtime=950,
                                    min_budget=50,
                                    max_budget=500)
```

For an epoch-based budget you simply have to add an extra parameter:

```py
autoPyTorch = AutoNetClassification(config_preset="tiny_cs", 
                                    budget_type='epochs', 
                                    min_budget=50, 
                                    max_budget=200, 
                                    num_iterations=5, 
                                    log_level='debug')
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
with open("logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)
```

