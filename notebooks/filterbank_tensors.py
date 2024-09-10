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
import ws_optimization.sw_dataset as sw

# %% Get 1D signal from dataloader
data_load = swd.SolarWindDataset(dir_path = 'C:/Users/jkobayashi/gh_repos/idea-lab-sw-isax/sw-data/nasaomnireader/1999')
signal_1d = data_load.__getitem__(10)

# %% Apply fourier transform using PyTorch FFT (should return 1D tensor)


# %% Create a 1D tensor with frequencies of triangle apexes


# %% Create a 2D that will hold the filtered spectrum for each filter 
# (Dimensions = n_filters x length(fourier_spectrum))


# %% Loop over apexes, multiply triangle frequencies and multiply by fourier spectrum
# Place result in 2D repository tensor


# %% Inverse Fourier transform on each row and add to get original signal