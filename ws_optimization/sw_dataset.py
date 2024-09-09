import torch
from torch.utils.data import Dataset
import datetime as dt
import cdflib

# version 2
import glob

# version 3
import pandas as pd
import numpy as np

# local imports
import os
_FILE_DIR = os.path.dirname( os.path.abspath(__file__))
_MODEL_DIR = os.path.dirname(_FILE_DIR)
os.chdir(_MODEL_DIR)
import fdl21.data.prototyping_metrics as pm

# %%
def omni_cdf_fname(data_dir,
                  year:int = 2000,
                  month:int = 1):
    
    fname = data_dir + f'/{year}/omni_hro_1min_{year}{month:02d}01_v01.cdf'
    return fname

# %% Get single cdf file (i.e. getitem = 1 month of data)
class SolarWindDataset(Dataset):
    def __init__(self,
                 fname,
                 parameter:str = 'flow_speed',):
        """[summary]
        Parameters
        ----------
        fnaame : str
            Path to the directory of cdf data files."""
        self.parameter = parameter
        self.files = glob.glob(fname+'/**',recursive=True)
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        self.data = cdflib.CDF(fname)
        return self.data[self.parameter], fname
