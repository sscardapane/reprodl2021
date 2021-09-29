# Reproducible Deep Learning
## Extra: Poetry
### Author: [Luca Maiano](https://github.com/lucamaiano)
[Poetry](https://python-poetry.org/)

Before going into this branch, please look at the main branch in order to understand the project details.
>⚠️ extra branches implement additional exercises created by the students of the course to explore additional libraries and functionalities.
> They can be read independently from the main branches. Refer to the original authors for more information.

&nbsp;

This is an extra branch of **exercise2_hydra_completed** that is intended to introduce Poetry, a tool for dependency management and packaging in Python. 

&nbsp;

## Goal
Learn how to integrate Poetry for easier dependency management! Poetry is a Python packaging and dependency management system initially released in 2018. It smoothly handles the dependencies and virtual environments of your projects and allows for easier reproducibility in teams.

## Step 1: installation
1. Install Poetry on your system. Poetry provides a custom installer that will install Poetry isolated from the rest of your system by vectorizing its dependencies.
If you are a Mac OS X or Linux user type:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

If you are on Windows:

```bash
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

2. To check your installation, open a new shell and type the following:

```bash
poetry --version
```

Check the [official documentation](https://python-poetry.org/docs/#installation) for further details.

&nbsp;

## Step 2: project setup

Poetry offers two ways to initialize a project: (i) the `new` command allows you to create a new project from scratch and will automatically initialize a folder with structured content; and (ii) `init` which we will use here since we want to add poetry to an existing project. Move with the terminal inside the folder of this repository and type:

```bash
poetry init
```

This command will guide you through creating your `pyproject.toml` config file. This will prompt few questions about the desired Python packages you want to install. You can press Enter to proceed with default options if you want to skip this part now. The *toml* file have a main section for all dependencies (used in both production and development environments), but you can also define a section that contains packages mainly used for development purposes only. This is the one of the advantages over other dependency management tools. Only one configuration file is required for both the production and development environments. Eventually, you will find your configuration file with a structure similar to this.

```toml
[tool.poetry]
name = "reprodl2021"
version = "0.1.0"
description = "Reproducible deep learning course."
authors = ["Luca Maiano <maiano@diag.uniroma1.it>"]

[tool.poetry.dependencies]
python = "^3.8"
torch = "^1.9.1"
torchaudio = "^0.9.1"
pytorch-lightning = "^1.4.8"
hydra-core = "^1.1.1"
hydra-colorlog = "^1.1.0"
matplotlib = "^3.4.3"
pandas = "^1.3.3"

[tool.poetry.dev-dependencies]
black = {version = "^21.9b0", allow-prereleases = true}

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

Once you have your dependencies and other configurations in a `pyproject.toml` file, you can install the dependencies by simply running:

```bash
poetry install
```

If you have never run the command before and there is also no `poetry.lock` file present, Poetry simply resolves all dependencies listed in your `pyproject.toml` file and downloads the latest version of their files. When Poetry has finished installing, it writes all of the packages and the exact versions of them that it downloaded to the `poetry.lock` file, locking the project to those specific versions. If you're running this command on an existing repository containing the *lock* file, you can specify to the command that you do not want the development dependencies installed by passing the `--no-dev` option.
Committing this file in git is important because it will ensure that whoever sets up the project is using the exact same dependency versions you are using. This will ensure that everything and everyone run on the same dependencies, which reduces the potential for bugs affecting only certain parts of distributions. I strongly advise you not to manually update the `poetry.lock` file manually. Let Poetry do its magic!

&nbsp;

## Step 3: using your virtual environment

By default, poetry creates a virtual environment in `{cache-dir}/virtualenvs` (`{cache-dir}\virtualenvs` on Windows). You can change the `cache-dir` value by editing the poetry config. Additionally, you can use the `virtualenvs.in-project` configuration variable to create virtual environment within your project directory.

There are two ways to run commands within this virtual environment.

###  Using poetry run

To run your script simply use `poetry run python train.py`. In this way there is no need to activate/deactivate the virtual environment.

###  Activating the virtual environment

The easiest way to activate the virtual environment is to create a new shell with `poetry shell`. To deactivate the virtual environment and exit this new shell type `exit`. To deactivate the virtual environment without leaving the shell use `deactivate`.

&nbsp;

## Poetry tips

### Add new packages

If you want to add a package to your environment, you can use the following command:

```bash
poetry add black
```

This will automatically add the package name and version to your `pyproject.toml` file and updates the `poetry.lock` accordingly. Poetry **add** takes care of all dependencies, and adds the package in the `[tool.poetry.dependencies]` section.

If you want to add a package to your development environment, you can simly pass a `--dev` option as below

```bash
ppoetry add --dev black
```

You also can specify a constraint when adding a package, like so:

```bash
poetry add "torch@^1.9.1"
```

If you want to get the latest version of an already present dependency you can use the special latest constraint:

```bash
poetry add torch@latest
```

### Remove packages

The remove command removes a package from the current list of installed packages.

```bash
poetry remove ipython
```

Again, you can use the `--dev` option to remove a package from the development dependencies.

### Updating packages to the latest versions

As mentioned above, the `poetry.lock` file prevents you from automatically getting the latest versions of your dependencies. If you want to force the installation of the latest versions, use the **update** command. This will fetch the latest matching versions (according to your `pyproject.toml` file) and update the lock file with the new versions. 

```bash
poetry update
```

If you just want to update a few packages and not all, you can list them as such:

```bash
poetry update torch torchaudio
```