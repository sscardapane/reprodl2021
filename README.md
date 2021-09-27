# Reproducible Deep Learning
## Extra: Python-Poetry
Author: Timur Obukhov

Poetry is an easy tool for managing dependency and packaging in Python. It allows to declare the libraries of project dependencies, and it will manage them. Poetry is packaging and managing dependencies using a single file called ```pyproject.toml```

## Installtion

Python-Poetry for windows installation instruction: 

```bash
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python -
```

> ⚠️ The previous get-poetry.py installer is now deprecated.

The installer installs the poetry tool to Poetry's bin directory. For Windows it is in:

```bash
%APPDATA%\Python\Scripts
```

If this directory is not on your PATH, you will need to add it manually if you want to invoke Poetry with simply ```poetry```.

It is also possible to install Poetry from a ```git``` repository by using ```--git``` option: 

```bash
python install-poetry.py --git https://github.com/python-poetry/poetry.git@master
```




## Use of Poetry
### Potery commands: 
It is possible to build, add dependencies, track and publish packages do so other monipulation with packages. 
Avaibale commnads for comand-line can be found here: [Comands for Potery](https://python-poetry.org/docs/cli/).


To use the trained model and sound recodnition the potery created the following folder structure: 

```bash
tim_ob
├── .python-version
├── pyproject.toml
├── poetry.lock
├── config.py
├── main.py
├── send_file.py
├── send_file_app.py
├── train_bis.py
├── dog_sound_test.wav
└── _pycache_
    ├── config.cpython-39.pyc
    ├── main.cpython-39.pyc
    ├── send_file.cpython-39.pyc
    ├── send_file_app.cpython-39.pyc
    └── train_bis.cpython-39.pyc
```


