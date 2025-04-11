import sys
sys.path.append('../')

from gears import PertData
import numpy as np

def load_adamson(seed=1):
  pert_data = PertData('../notebook-scGPT/data') # specific saved folder
  pert_data.load(data_name = 'adamson') # specific dataset name
  pert_data.prepare_split(split = 'simulation', seed = seed) # get data split with seed
  pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader
  return pert_data


def load_k562(seed=1):
  pert_data = PertData('../notebook-scGPT/data') # specific saved folder
  pert_data.load(data_name = 'replogle_k562_essential') # specific dataset name
  pert_data.prepare_split(split = 'simulation', seed = seed) # get data split with seed
  pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader
  return pert_data


def load_rpe1(seed=1):
  pert_data = PertData('../notebook-scGPT/data') # specific saved folder
  pert_data.load(data_name = 'replogle_rpe1_essential') # specific dataset name
  pert_data.prepare_split(split = 'simulation', seed = seed) # get data split with seed
  pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader
  return pert_data

