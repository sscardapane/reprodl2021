# Reproducible Deep Learning
## Extra: Ax Platform in PyTorch for hyperparameters tuning using Bayesian optimization 
### Author: [gditeodoro](https://github.com/gditeodoro)
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)] 


>⚠️ extra branches implement additional exercises created by the students of the course to explore additional libraries and functionalities.
> They can be read independently from the main branches. Refer to the original authors for more information.


This is an extra branch of the **exercise 1** to tune hyperparameters with Bayesian optimization. Adaptive experimentation is the machine-learning guided process of iteratively exploring a (possibly infinite) parameter space in order to identify optimal configurations in a resource-efficient manner, avoiding exhausting random search and grid search processes.

## Goal
Implement Bayesian optimization to tune hyperparameters and find automatically the optimal ones in PyTorch using [Ax](https://ax.dev/).
The details of Bayesian optimization can be found on the [Ax website](https://ax.dev/docs/bayesopt.html).

## Prerequisites 

Install requirements: 
- Install the ax platform 
```bash
pip3 install ax-platform
```
## Instructions 

1. Import the `optimize`, `train`, and `evaluate` functions from the Ax package: 
```python

from ax.service.managed_loop import optimize
from ax.utils.tutorials.cnn_utils import train, evaluate

```
2. Open the script `train.py` that has to be modified.
3. Modify the `train()` function in a way that:
    -  the model is initialized and the network is ready-to-train. The parameterization argument is a dictionary containing the hyperparameters.
    -  the `train()` function is called by the Bayesian optimizer on every run. A new set of hyperparameters is generated in the parameterization by the optimizer, this set is passed to this function and the returned evaluation results are analyzed. 
    
```python
# constructing a new training data loader allows us to tune the batch size and the learning rate 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=parameterization.get("batchsize", 8), shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=parameterization.get("batchsize", 8))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=parameterization.get("batchsize", 8))
audionet = AudioNet(lr=parameterization.get("lr", 1e-3))

# return the evaluation results
return evaluate(
        net=audionet,
        data_loader=test_loader,
        dtype=dtype,
        device=device,
    )
```
4. Optimize, specifing the hyperparameters you want to sweep across and pass that to Ax’s `optimize()` function:
```python
best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "batchsize", "type": "range", "bounds": [8, 32]}
        ],

        evaluation_function=train,
        objective_name='accuracy',
    )
```
