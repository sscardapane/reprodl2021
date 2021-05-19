# Reproducible Deep Learning
## Exercise 1: Git & Scripting
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)] [[Slides](https://docs.google.com/presentation/d/1_AYIcCyVI59QiiXqU4Sn7VzwtVyfqv-lG36EPFzeSdY/edit?usp=sharing)]

## Objectives for the exercise

- [ ] Experimenting with Git branches.
- [ ] Turning a notebook into a runnable script.

See the completed exercise:

```bash
git checkout exercise1_git_completed
```

## Prerequisites

1. Uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. Run *Initial Notebook.ipynb* to see an example of training.

## Instructions

The aim of this exercise is to get some familarity with Git branches and Python scripts. You are tasked with turning the [training notebook](Initial%20Notebook.ipynb) into a runnable Python script, working on a separate Git branch, and merging the result at the end.

1. Start by initializing and moving to an experimental branch:

```bash
git branch experimental_branch
git checkout experimental_branch
```

2. Convert the notebook into a Python script by running `nbconvert`:

```bash
jupyter nbconvert --to script --output "train" "Initial Notebook.ipynb"
```

> :speech_balloon: The command has [several useful flags](https://nbconvert.readthedocs.io/en/latest/config_options.html) to simplify the conversion (e.g., check `TemplateExporter.exclude_markdown`).

3. Reorganize the script so that it is runnable from terminal:
   * Remove all instructions that are not required for training;
   * Put all training instructions inside a new `train()` function.

4. Add a [top-level instruction](https://docs.python.org/3/library/__main__.html) to run the module as a script:

```python
if __name__ == "__main__":
    train()
```

5. Create a [.gitignore file](https://git-scm.com/docs/gitignore) to ignore the *data* and *lightning_logs* folders.
6. Remove the notebook, and check that the training script is working correctly:

```bash
python train.py
```

7. Merge the experimental branch into the main one, and delete the experimental branch:

```bash
git checkout main
git merge experimental_branch
git branch -d experimental_branch
```

Congratulations! You have concluded the first move to a reproducible deep learning world. :nerd_face:

Move to the next exercise:

```bash
git checkout exercise2_hydra
```
