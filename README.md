# Reproducible Deep Learning
## Exercise 6: Continuous Integration
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)]

## Objectives for the exercise

- [ ] Unit testing the code.
- [ ] Automatic formatting with [Black](https://github.com/psf/black).
- [ ] Integrating checks in pre-commit hooks.
- [ ] Adding a GitHub Action to automate the process.

See the completed exercise:

```bash
git checkout exercise6_hooks_completed
```

## Prerequisites

1. Complete (at least) [exercise 2](https://github.com/sscardapane/reprodl2021/tree/exercise2_hydra).
   
2. Install `black` and `nose`:

```bash
pip install black nose
```

## Preliminary: testing and formatting

The purpose of this exercise is to integrate two ways of checking the code correctness: **formatting** and **testing**. 

### Step 1: Add a unit test

To start, create a `test_audionet.py` file with at least one test function:

```python
def testAudioNet():
  ...
  assert # Test something about the model
```

You can read the [Hydra documentation](https://hydra.cc/docs/next/advanced/unit_testing/) for information on how to load the configuration inside the unit test. If you need an idea for the unit test, PyTorch Lightning has a useful [overfit on batch](https://pytorch-lightning.readthedocs.io/en/latest/common/debugging.html#make-model-overfit-on-subset-of-data) function that provides a quick test for the correctness of the model. 

Launch the unit test:

```bash
nosetests
```

### Step 2 - Add formatting

[Black](https://github.com/psf/black) provides a powerful way to automatically format any code according to well-defined style guides.

First, check the adherence of your code to the Black style:

```bash
black --diff --check train.py test_audionet.py
```

Then, try to apply the changes:

```bash
black train.py test_audionet.py
```

> :speech_balloon: Black is an **opinionated** formatting tool. If you do not like the resulting formatting, feel free to skip this step.

## Add a Git hook

Our challenge now is to ensure that testing and formatting are applied to every commit we make. A simple way to ensure this is to add a [pre-commit Git hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks).

> :speech_balloon: Because this is a didactic exercise, we implement the hook from scratch. For a realistic use case, consider [pre-commit](https://pre-commit.com/).

First, write a `pre-commit` script that launches our two testing routines. A small template is provided below:

```bash
#!/bin/sh

# TODO: Launch the Black checker here
...

# If the command does not return 0 ('correct'), we exit the commit
if [ $? -ne 0 ]; then
	echo "Code is not formatted correctly!"
	exit 1
fi

# TODO: Repeat a second time for the unit tests
```

Move the file to `.git/hooks/`, and try launching a commit with poorly formatted code (it should fail). Then, repeat after executing the Black formatter.

## Adding a GitHub Action

The Git hook does its job, but it is limited, especially because it runs locally. **Continuous integration** platforms (like CircleCI, Travis CI, ...) provide a way of integrating these checks when a user pushes the code to the remote repository.

For this exercise, we use GitHub Actions: read the [quickstart](https://docs.github.com/en/actions/quickstart) before proceeding.

Create a `push-workflow.yml` file inside a `.github/workflows/` folder. The workflow shoud: (i) install all dependencies for our code; (ii) execute the unit tests; (iii) check adherence of the code to the Black style.

An incomplete template is provided below:

```yaml
name: push-workflow
on: [push]
jobs:
  build: # Build the environment
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2 # Checkout the repository
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          sudo apt-get install -y libsndfile1-dev
      - name: Run all unit tests 
        run: ... # TODO: Add code here to run the unitests
      # TODO: Add another step to run Black
```

Push the new file to GitHub, then try committing additional code. You can visualize the running workflows from the Actions tab of the GitHub repository.

Congratulations! You have concluded another move to a reproducible deep learning world. :nerd_face:

This exercise concludes the course. I hope you enjoyed it! Feel free to contact me or open an issue if you want to provide any feedback.
