#!/usr/bin/env python
# coding: utf-8

import torch, torchaudio
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
logger = logging.getLogger(__name__)

'''
This code should be used to collect data iteratively, whether from the web or perhaps generated in some way.
Here, however, I have created a dummy code that simply enlarges a target dataset by taking pieces of the original ESC-50 dataset.
'''

def retrieve_data(path, target_dataset,data_size):

    original_dataset_path = path / Path('meta/esc50.csv')
    original_dataset = pd.read_csv(original_dataset_path) #load original dataset

    print(original_dataset.shape)
    print(target_dataset.shape)

    original_index = original_dataset.index
    target_index = target_dataset.index

    not_retrieved_index = original_index.difference(target_index)

    print(len(not_retrieved_index))

    retrieved_index = np.random.choice(not_retrieved_index,data_size)

    print("RETRIEVED:", retrieved_index)

    return original_dataset.loc[retrieved_index]

def add_data(retrieved_dataset,target_dataset):

    return target_dataset.append(retrieved_dataset,ignore_index=False, sort=True)


@hydra.main(config_path='configs', config_name='default')
def main(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.
    
    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))
    
    # We recover the original path of the dataset:
    path = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.path)

    
    #Retrieve data
    #Dummy code: take chunks from the original ESC-50 dataset

    target_dataset_path = path / Path('meta/extra_esc50.csv') #path of target data
    if os.path.isfile(target_dataset_path): #if target dataset exists
        target_dataset = pd.read_csv(target_dataset_path) #load target dataset
    else:
        target_dataset = pd.DataFrame() #create empty dataset
    
    retrieved_dataset = retrieve_data(path, target_dataset,cfg.data.retrieval_size) #retrieve data

    target_dataset = add_data(retrieved_dataset,target_dataset) #add data

    target_dataset.to_csv(target_dataset_path, index=False) #save data
    print("SAVED")


if __name__ == "__main__":
    main()
