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

# %% Get specific data point (i.e. getitem = 1 data point)
class SolarWindDataset(Dataset):
    def __init__(self,
                 fname,
                 parameter:str = 'flow_speed',):
        """[summary]
        Parameters
        ----------
        fname : str
            Requires the full path to the specific cdf file to read."""
        cdf_file = cdflib.CDF(fname)
        self.parameter = parameter
        self.data = cdf_file[parameter]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
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

# %% More similar to prototyping metrics
class SolarWindDatasetOMNI(Dataset):
    def __init__(self,
                 data_dir,
                 year:int = 2000,
                 month:int = 1,
                 parameter:str = 'flow_speed',):
        """[summary]
        Parameters
        ----------
        data_dir : str
            Path to the data directory. (OMNI data)"""
        
        fname = omni_cdf_fname(data_dir,year,month)
        cdf_file = cdflib.CDF(fname)
        
        df = {}
        df[parameter] = cdf_file[parameter]
        dates = pm.convert_OMNI_EPOCH(cdf_file)
        
        self.data = pd.DataFrame(df,index=pd.DatetimeIndex(dates)).sort_index()
        self.data.index = self.data.index.round('min')

        self.year = year
        self.month = month
        self.parameter = parameter
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:tuple = (1,1,1)):
        idx = dt.datetime(self.year,self.month,day=idx[0],hour=idx[1],minute=idx[2])
        return self.data.loc[idx]

