## Ray Tune: distributed hyperparameter tuning

Ray Tune is a tool for hyperparameters tuning in a local or distributed fashion. It is integrated in TensorBoard and easy to add in a PyTorch code.

### Local set-up
As reported in the [documentation](https://docs.ray.io/en/latest/installation.html), Ray fully supports MacOS and Linux. Windows requires Visual C++ dependencies, check it [here](https://docs.ray.io/en/latest/installation.html#windows-support). 

To install Ray Tune follow:

```bash
pip install -U ray
```

In your code, add imports.

```python
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
```

The information passing between your model and Tune is performed by the `tune.report` method.
Here an example of validation loss and accuracy:

```python
tune.report(loss=validation_loss, accuracy=validation_accuracy)
```

The hyperparameter grid is a Python dictionary:

```python
config = {"lr": tune.loguniform(1e-4, 1e-1),
          "batch_size": tune.choice([32, 64])}
```

Finally, the search is performed through `tune.run` in which it is possible to specify the resources allocated to perform the search.

```python
result = tune.run(
    partial(train_cifar, data_dir=data_dir),
    resources_per_trial={"cpu": 8, "gpu": 2},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=True)
```

During training, if TensorBoard is installed, hyperparameters are automatically tracked and visualized in TensorBoard:

```bash
tensorboard --logdir ~/ray_results
```
