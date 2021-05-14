# Reproducible Deep Learning
## Exercise 5: Managing experiments with Weight & Biases
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)] [[Weight & Biases website](https://wandb.ai/)]

## Objectives for the exercise

- [ ] Logging the results of experiments using W&B.
- [ ] Optimizing hyper-parameters using W&B Sweeps.

See the completed exercise:

```bash
git checkout exercise5_wandb_completed
```

## Prerequisites

1. Complete (at least) [exercise 2](https://github.com/sscardapane/reprodl2021/tree/exercise2_hydra).
2. Create an account on [Weight & Biases](https://wandb.ai/).

> :speech_balloon: As an alternative, you can use a local [W&B server](https://hub.docker.com/r/wandb/local) deployed with Docker.

3. Install the `wandb` client:

```bash
pip install wandb
```

## Step 1 - Add W&B integration

Now that we have tracked everything (code, data, environment), we can start performing experiments with our code, and we need an efficient way of keeping track of all the results. Before starting, read the [W&B Quickstart](https://docs.wandb.ai/quickstart).

1. From a terminal, login into wandb:

```bash
wandb login
```

2. From inside the code, initialize the client:

```python
import wandb
wandb.init(project="reprodl")
```

3. Change the default logger in PyTorch Lightning:

```python
wandb_logger = pl.logger.WandbLogger()
trainer = pl.Trainer(..., logger=wandb_logger)
```

Launch a training script, and check the resulting logs on the [W&B dashboard](https://wandb.ai/).

## Step 2 - Fine-tuning hyperparameters

With an experiment manager configured, the aim is now to launch as many training iterations as possible to find the optimal combination of hyper-parameters. We will do this using [W&B Sweeps](https://docs.wandb.ai/guides/sweeps). Read the [quickstart](https://docs.wandb.ai/guides/sweeps/quickstart) before continuing.

> :speech_balloon: Hydra has a number of [sweepers](https://hydra.cc/docs/plugins/ax_sweeper/) already configured, which can be run almost immediately. For this exercise, we are using a sweeper that is not integrated to make the implementation more interesting.

1. Start by creating a `sweep.yaml` file with the hyper-parameter configuration. You can take inspiration from [here](https://docs.wandb.ai/guides/sweeps/quickstart#2-configure-your-sweep), and read the full configuration [here](https://docs.wandb.ai/guides/sweeps/configuration). 

```yaml
program: train.py
method: bayes
metric:
  name: val_acc
  goal: maximize
parameters:
  sample_rate: [2000, 4000, 8000]
  # Define other hyper-parameters here...
```

2. By default, the sweeper will call our program as `train.py -sample_rate=4000 ...`, which is conflicting with Hydra. You can follow this guide to [remove argument parsing](https://docs.wandb.ai/guides/sweeps/configuration#examples-5) altogether.

3. Now for the difficult part: `wandb.init()` requires a default configuration, which is updated if the script has been called from the sweeper. In the training script, initialize the configuration using the defaults in our `default.yaml`, then use the parsed configuration to update any value, e.g.:

```python
# Default configuration is taken from cfg
wandb.init(config={ 
  sample_rate = cfg.data.sample_rate, ... 
})

# Get the (possibly updated) configuration
cfg.data.sample_rate = wandb.config.sample_rate
```

4. Initialize and launch the sweep:

```bash
wandb sweep sweep.yaml
wandb agent <xxx>
```

From the web dashboard, you can check the running sweep from the corresponding tab.

Congratulations! You have concluded another move to a reproducible deep learning world. :nerd_face:

Move to the next exercise, or check some additional optional activities below:

```bash
git checkout exercise6_hooks
```
