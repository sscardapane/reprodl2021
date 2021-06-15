#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from crontab import CronTab

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path='configs', config_name='default')
def schedule(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.
    
    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))
    
    path = Path(hydra.utils.get_original_cwd())

    print(path)

    your_python_path = cfg.cron.python_path
    your_username = cfg.cron.username

    my_crontab = CronTab(your_username)

    print("CURRENT JOBS")
    for job in my_crontab:
        print(job)
        if cfg.cron.clean:
            my_crontab.remove(job)
            print("REMOVED\n")
        
    if not cfg.cron.stop:
        for py_name,cron_params in cfg.cron.py_cmds.items():

            #command to enter working directory
            cwd_cmd = "cd "+str(path)

            #python command
            py_cmd = your_python_path + "python "+str(path)+"/" + py_name+".py"
            print(cwd_cmd + "; " + py_cmd)
            cmd = cwd_cmd + "; " + py_cmd

            '''
            #check if command is running
            for i in my_crontab.find_command(cmd):
                running_cmd = str(i)
                print(running_cmd)
                if cmd in running_cmd:
                    time_info = running_cmd.split(cmd)[0].strip()
                    print(time_info)
                    
                    print("EXISTS")
                else:
                    print("NO")
            '''

            job = my_crontab.new(command = cmd)

            print(cmd)

            for k,v in cron_params.items():
                for k2,v2 in v.items():
                    print(k,k2,v2)
                    getattr(getattr(job,k),k2)(v2)
        

    my_crontab.write()

if __name__ == "__main__":
    schedule()
