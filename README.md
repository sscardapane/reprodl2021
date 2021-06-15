# Reproducible Deep Learning
## Extra: Python-Crontab
### Author: [siciliano-diag](https://github.com/siciliano-diag)

Before starting, read the main documentation for [Python-Crontab](https://gitlab.com/doctormo/python-crontab/) from the Python-Crontab project website.

> :warning: **extra** branches implement additional exercises created by the students of the course to explore additional libraries and functionalities. They can be read independently from the main branches. Refer to the original authors for more information.

&nbsp;


## Prerequisites

1. Uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. Install requirements:

```bash
pip install python-crontab
```

&nbsp;


## Goal

The aim of this exercise is to understand how to set cronjobs in a simple way, mixing the functionalities of python-crontab and hydra.


&nbsp;

## Instructions



1. Modify the path to the dataset in `train.py` script to `meta/extra_esc50.csv`.

```python
self.csv = pd.read_csv(path / Path('meta/extra_esc50.csv'))
```

In this way we can use a dataset that will be built by a cron job.


&nbsp;

2. Set the `cron` section in the `default.yaml` file.

> :speech_balloon: This step is just to explain how to do it. For the first try, can be left as it is.

The `cron` section in the `default.yaml` is used by `scheduler.py` to initialize cron jobs.
in particular, this is the list of parameters and their purpose:
    - `username`: the machine username that will run the cron jobs
    - `python_path`: specify the path to python; without this, there may be problems such as "Module not found".
    - `clean`: If True, the cronjobs in the crontab will be removed
    - `stop`: If True, the new cronjobs will not be added to the crontab
    - `py_cmds`: the list of python files to run, with the frequency at which they will be executed
        - <NAME_OF_PY_FILE>: name of the .py file to run
            - <time_type>: name of the time specification as in the `python-crontab` module (e.g. `minute`, `hour`)
                - <time_value>: value for the time specification (e.g. `4`, `MON`)

&nbsp;

3. Run `scheduler.py`.

```python
python scheduler.py
```

It will follow the specifications in the `default.yaml` file to set the cron jobs.

If the scheduler is itself in the `py_cmds` of the configuration, it will run itself continuosly. In this way, you may modify the configuration file any time you want, and the scheduler will automatically update the cron jobs.

> :speech_balloon: Be careful, as including the scheduler in the cron jobs may start running the other programs in the background and you may not notice.

&nbsp;


4. Watch the cron jobs executing.

The `scheduler` is set to run `data_retrieval.py` every 10 minutes, and the `train.py` every 2 hours.
`data_retrieval.py` is a dummy code to simulate a situation where is is needed to retrieve data continuously from some source and add to the main dataset. In this case, it will simply select 50 new rows of the `ESC-50.csv` and it will add them to `extra_ESC-50.csv` (if it does not exist, it will be created).

