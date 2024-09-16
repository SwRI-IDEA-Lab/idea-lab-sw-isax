# %% libraries
import torch
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

# local imports
import os
_FILE_DIR = os.path.dirname( os.path.abspath(__file__))
_MODEL_DIR = os.path.dirname(_FILE_DIR)
os.chdir(_MODEL_DIR)
import fdl21.data.prototyping_metrics as pm
import ws_optimization.sw_dataset as swd
import fdl21.data.build_filterbanks as fb
# %% [markdown] ==================================================================================================
# ## Get 1D signal from dataloader 
# %% 1D signal from dataloader ----------------------------------------------------------------------------------
data_load = swd.SolarWindDataset(dir_path = 'C:/Users/jkobayashi/gh_repos/idea-lab-sw-isax/sw-data/nasaomnireader/1999')
signal_1d = data_load.__getitem__(10)
signal_1d

# %% [markdown] ==================================================================================================
# ## Apply fourier transform using PyTorch FFT 
# (should return 1D tensor)
 # %% Fourier transform ------------------------------------------------------------------------------------------
sig_fft = torch.fft.fft(signal_1d)# - torch.mean(signal_1d))
# print(sig_fft)
# sig_fft = torch.real(sig_fft)
print(sig_fft)

# %% [markdown] ==================================================================================================
# ## Create a 1D tensor with frequencies of triangle apexes 
# %% Filterbanks (from local filterbank module)------------------------------------------------------------------
fbank = fb.filterbank(restore_from_file='C:/Users/jkobayashi/gh_repos/idea-lab-sw-isax/data/filterbanks/fb_0.01_1.66_3.33_5.0_DC_HF.pkl')
fbank.visualize_filterbank()

# %% Tensors of matrix and frequencies---------------------------------------------------------------------------
mat = torch.tensor(fbank.fb_matrix)     # 2D tensor of filterbank matrix (triangles)
freq = torch.tensor(fbank.fftfreq)      # 1D tensor of frequencies


# %% [markdown] ==================================================================================================
# ## Create a 2D that will hold the filtered spectrum for each filter  
# (Dimensions = n_filters x length(fourier_spectrum))

# %%
all_filtered_signals = torch.empty(mat.shape[0], signal_1d.shape[0])

# %% [markdown] ==================================================================================================
# ## Loop over apexes, multiply triangle frequencies and multiply by fourier spectrum
# Place result in 2D repository tensor 
# 
# Inverse Fourier transform on each row and add to get original signal


# %%
sample_freq = torch.fft.fftfreq(signal_1d.shape[0],d=1/60)
total_sig = torch.zeros(signal_1d.shape[0])
for i,bank in enumerate(mat):
    filter = torch.tensor(np.interp(np.abs(sample_freq),freq,bank,left=None,right=None,period=None))
    filteredYF = sig_fft.clone()
    filteredYF = filteredYF*filter

    filtered_sig = torch.real(torch.fft.ifft(filteredYF))

    all_filtered_signals[i] = filtered_sig
    total_sig += filtered_sig

# %%
fig, axes = plt.subplots(figsize=(8,5))
axes.plot(signal_1d,linewidth=5,label='Original Signal')
axes.plot(total_sig,label='Total Filtered Signal')
axes.legend()
plt.show()