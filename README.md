# Reproducible Deep Learning
## Exercise 2: Configuration with Hydra
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)]

## Objectives for the exercise

- [ ] Adding Hydra support for configuration.
- [ ] Experimenting with (colored) logging inside the script.

See the completed exercise:

```bash
git checkout exercise2_hydra_completed
```

## Prerequisites

1. Uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. If this is your first exercise, run *train.py* to check that everything is working correctly.
3. Install [Hydra](https://github.com/facebookresearch/hydra):

```bash
pip install hydra-core
```

## Instructions

The aim of this exercise is to move all configuration for the training script inside an external configuration file. This simple step dramatically simplifies development and reproducibility, by providing a single entry point for most hyper-parameters of the model.

1. Go through the training script, and make a list of all values that can be considered hyper-parameters (e.g., the **learning rate** of the optimizer, the **sampling rate** of the dataset, the **batch size**, ...).

2. Prepare a `configs/default.yaml` file collecting all hyper-parameters. We suggest the following (macro) organization, but you are free to experiment:

```yaml
data:
    # All parameters related to the dataset
model:
    # All parameters related to the model
    optimizer:
        # Subset of parameters related to the optimizer
trainer:
    # All parameters to be passed at the Trainer object
```

> :speech_balloon: Check the [basic example](https://hydra.cc/docs/intro/#basic-example) in the Hydra website for help in creating the configuration file.

3. Decorate the `train()` function to accept the configuration file:

```python
@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):
    # ...
```

4. Use the `cfg` object to set all hyper-parameters in the script.
   * Read the [OmegaConf](https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#access-and-manipulation) documentation to learn more about accessing the values of the `DictConfig` object.
   * `LightningModule` [accepts dictionaries to set hyper-parameters](https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html).
   * The `Trainer` object requires key-value parameters, but you can easily solve this with:
        ```python
        trainer.fit(**cfg.trainer)
        ```
    * Hydra will run your script inside a folder which is dynamically created at runtime. To load the dataset, you can use `hydra.utils.get_original_cwd()` to [recover the original folder](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory#original-working-directory).

If everything went well, you should be able to experiment a little bit with Hydra configuration management:

- Run a standard training loop: 
  
  ``` python train.py ```
- Dynamically change the batch size (modify according to your YAML file): 
  
  ``` python train.py data.batch_size=4 ```

- Add new flags for the trainer on-the-fly:
  
  ``` python train.py +trainer.fast_dev_run=True ```

- Remove a flag from the configuration:
   
  ``` python train.py ~trainer.gpus ```
  
- Change the output directory:
   
  ``` python train.py hydra.run.dir="custom_dir" ```

Congratulations! You have concluded another move to a reproducible deep learning world. :nerd_face:

Move to the next exercise, or check some additional optional activities below:

```bash
git checkout exercise3_dvc
```

### Optional: add some logging

Using Hydra, we can easily add any amount of logging into the application, that is automatically saved inside the dynamically generated folders.

Start by importing the logging function:
  
```python 
import logging
logger = logging.getLogger(__name__)
```

You can use the logger to save some important information, debug messages, errors, etc. You will find the log inside the `train.log` file in each generated folder. For example, you can log the initial configuration of the training script:

```python 
from omegaconf import OmegaConf
logger.info(OmegaConf.to_yaml(cfg))
```

Finally, you can color the logging information to make it more readable on terminal. Hydra has a number of interesting plugins, including a [colorlog](https://hydra.cc/docs/plugins/colorlog) plugin. First, install the plugin:

```bash
pip install hydra_colorlog
```

Then, add these instructions inside your configuration file:

```yaml
defaults:
  - hydra/job_logging: colorlog
  - hydra/hydra_logging: colorlog
```

Run again the scripts above to see the colored output.
