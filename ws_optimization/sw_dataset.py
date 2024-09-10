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

# %% Get single cdf file (i.e. getitem = 1 month of data)
class SolarWindDataset(Dataset):
    def __init__(self,
                 dir_path:str,
                 parameter:str = 'flow_speed',):
        """[summary]
        Parameters
        ----------
        fnaame : str
            Path to the directory of cdf data files."""
        self.parameter = parameter
        self.files = glob.glob(dir_path+'/**',recursive=True)
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        self.data = cdflib.CDF(fname)
        return torch.tensor(self.data[self.parameter])
