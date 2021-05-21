# Reproducible Deep Learning
## PhD Course in Data Science, 2021, 3 CFU
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)]

This practical PhD course explores the design of a simple *reproducible* environment for a deep learning project, using free, open-source tools ([Git](https://git-scm.com/), [DVC](http://dvc.org/), [Docker](https://www.docker.com/), [Hydra](https://github.com/facebookresearch/hydra), ...). The choice of tools is opinionated, and was made as a trade-off between practicality and didactical concerns.

## Local set-up

The use case of the course is an audio classification model trained on the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. To set-up your local machine (or a proper virtual / remote environment), configure [Anaconda](https://www.anaconda.com/products/individual), and create a clean environment:

```bash
conda create -n reprodl; conda activate reprodl
```

> ⚠️ For an alternative setup without Anaconda, see [issue #2](https://github.com/sscardapane/reprodl2021/issues/2).

Then, install a few generic prerequisites (notebook handling, Pandas, …):

```bash
conda install -y -c conda-forge notebook matplotlib pandas ipywidgets pathlib
```

Finally, install [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). The instructions below can vary depending on whether you have a CUDA-enabled machine, Linux, etc. In general, follow the instructions from the websites.

```bash
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c conda-forge
conda install -y pytorch-lightning -c conda-forge
```

This should be enough to let you run the [initial notebook](https://github.com/sscardapane/reprodl2021/blob/main/Initial%20Notebook.ipynb). More information on the use case can be found inside the notebook itself.

> :warning: For Windows only, install a [backend for torchaudio](https://pytorch.org/audio/stable/backend.html):
> ```bash
> pip install soundfile
> ```

### Additional set-up steps

The following steps are not mandatory, but will considerably simplify the experience.

1. If you are on Windows, install the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10). This is useful in a number of contexts, including Docker installation.
2. We will use Git from the command line multiple times, so consider enabling [GitHub access with an SSH key](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh).
3. We will experiment with Docker reproducibility on the [Sapienza DGX environment](https://www.uniroma1.it/sites/default/files/field_file_allegati/presentazione_ga_13-05-2019_sgiagu.pdf). If you have not done so already, set-up your access to the machine.

## Organization of the course

<p align="center">
<img align="center" src="https://github.com/sscardapane/reprodl2021/blob/main/reprodl_overview.png" width="500" style="border: 1px solid black;">
</p>

The course is split into **exercises** (e.g., adding DVC support). The material for each exercise is provided as a Git branch. To follow an exercise, switch to the corresponding branch, and follow the README there. If you want to see the completed exercise, add *_completed* to the name of the branch. Additional material and information can be found on the [main website](https://www.sscardapane.it/teaching/reproducibledl/) of the course.

**List of exercises**:

- [x] Experimenting with Git, branches, and scripting (*exercise1_git*).
- [x] Adding Hydra configuration (*exercise2_hydra*).
- [x] Versioning data with DVC (*exercise3_dvc*).
- [x] Creating a Dockerfile (*exercise4_docker*).
- [x] Experiment management with Weight & Biases (*exercise5_wandb*). 
- [x] Unit testing and formatting with continuous integration (*exercise6_hooks*).

### An example

If you want to follow the first exercise, switch to the corresponding branch and follow the instructions from there:

```bash
git checkout exercise1_git
```

If you want to see the completed exercise:

```bash
git checkout exercise1_git_completed
```

You can inspect the commits to look at specific changes in the code:

```bash
git log --graph --abbrev-commit --decorate
```

If you want to inspect a specific change, you can checkout again using the ID of the commit.

### Contributing

Thanks to [Jeroen Van Goey](https://github.com/BioGeek) for the error hunting. Feel free to open a pull request if you have suggestions on the current material or ideas for some extra exercises (see below). 

> ⚠️ Because of the sequential nature of the repository, changing something in one of the initial branches might trigger necessary changes in all downstream branches.

### Extra material (students & more)

**Extra** branches contain material that was not covered in the course (e.g., new libraries for hyper-parameter optimization), implemented by the students for the exam. They can be read independently from the main branches. Refer to the original authors for more information.

| Author | Branch | Content |
| ------------- | ------------- |------------- |
| [OfficiallyDAC](https://github.com/OfficiallyDAC) | [extra_optuna](https://github.com/sscardapane/reprodl2021/tree/extra_optuna) | Fine-tuning hyper-parameters with [Optuna](https://optuna.readthedocs.io/en/latest/installation.html). |
| [FraLuca](https://github.com/FraLuca) | [extra_torchserve](https://github.com/sscardapane/reprodl2021/tree/extra_torchserve) | Serving models with [TorchServe](https://pytorch.org/serve/) |

### Advanced reading material

If you liked the exercises and are planning to explore more, the new edition of [Full Stack Deep Learning](https://fullstackdeeplearning.com/) (UC Berkeley CS194-080) covers a larger set of material than this course. Another good resource (divided in small exercises) is the [MLOps](https://github.com/GokuMohandas/mlops) repository by Goku Mohandas. [lucmos/nn-template](https://github.com/lucmos/nn-template) is a fully-functioning template implementing many of the tools described in this course.
