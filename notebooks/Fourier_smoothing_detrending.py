# %%
# %% libraries

import cdflib

from tqdm import tqdm
from pathlib import Path
import pandas as pd
from numpy import abs, append, arange, insert, linspace, log10, round, zeros
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from multiprocessing import Pool
from scipy.stats import norm

import dill as pickle

from astropy.time import Time 

import spiceypy as spice
from mpl_toolkits.mplot3d import Axes3D 


from tslearn.piecewise import PiecewiseAggregateApproximation

from astropy.time import Time 

import spiceypy as spice
from mpl_toolkits.mplot3d import Axes3D 


from pyfilterbank import melbank
from scipy import fft

import datetime as dt
import os,sys

_FILE_DIR = os.path.abspath("")
_SRC_DIR = os.path.dirname(_FILE_DIR)
sys.path.append(_SRC_DIR)

# local imports
import fdl21.data.prototyping_metrics as pm
import fdl21.utils.time_chunking as tc
import fdl21.data.build_filterbanks as fb

# %% [markdown]
# # Prepare test data

# %%
# %% Get data

year = '2019'
month = '05'
test_cdf_file_path =_SRC_DIR+fb._OMNI_MAG_DATA_DIR+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

mag_df = fb.get_test_data(fname_full_path=test_cdf_file_path)
cols = ['BY_GSE']
mag_df=mag_df[cols]
mag_df

# %%
# %% Prepare FT of test data for Fourier applications

cadence = dt.timedelta(seconds=60)

mag_df.sort_index(inplace=True)
mag_df.interpolate(method='index', kind='linear',limit_direction='both',inplace=True)
df_index=pd.date_range(start=mag_df.index[0], end=mag_df.index[-1], freq=cadence)

sig_fft_df = fft.rfftn(mag_df - mag_df.mean(),axes=0)

# %% [markdown]
# # Theoretical Frequency Response
# 
# **(Formula from [Ch. 15 of *Digital Signal Processing Textbook*](https://www.dspguide.com/CH15.PDF)**)
# 
# Frequency response of an $M$ point moving average filter. The frequency, $f$, runs between $0$ and $0.5$. For $f = 0$, use $H[f] = 1$
# 
# $$H[f] = \frac{\sin(\pi f M)}{M\sin(\pi f)}$$

# %%
# %% Ch. 15 Formula
def moving_avg_freq_response(f,window=dt.timedelta(minutes=3000),cadence=dt.timedelta(minutes=1)):
    n = int(window.total_seconds()/cadence.total_seconds())
    numerator = np.sin(np.pi*f*n)
    denominator = n*np.sin(np.pi*f)
    return abs(numerator/denominator)

# %% [markdown]
# # Smoothing

# %%
smooth_window = dt.timedelta(seconds=18000)

# %%
# Build theoretical frequency response 
data_len = mag_df.shape[0]  
freq_spectrum = np.linspace(0.0001,data_len/2,(data_len//2)+1)
smooth_FR_theory = moving_avg_freq_response(f=freq_spectrum,
                                        window=smooth_window,
                                        cadence=cadence)

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Smoothing Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum,smooth_FR_theory)
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum,smooth_FR_theory)
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')

# %% [markdown]
# # Detrending

# %%
detrend_window = dt.timedelta(seconds=21000)

# %%
FR_theory = moving_avg_freq_response(f=freq_spectrum,
                                        window=detrend_window,
                                        cadence=cadence)
detrend_FR_theory = 1 - FR_theory

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Detrending Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum,detrend_FR_theory)
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum,detrend_FR_theory)
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')

# %% [markdown]
# # 

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(15,15))
fig.suptitle("Frequency Responses",fontsize=18)
FR = [smooth_FR_theory,detrend_FR_theory,smooth_FR_theory-detrend_FR_theory]
for i in range(3):
    axes[i][0].plot(freq_spectrum,FR[i])
    axes[i][1].plot(freq_spectrum,FR[i])
    axes[i][1].set_xlim(0,100)
axes[0][0].set_title('Full Spectrum')
axes[0][1].set_title('Zoomed in')
axes[2][0].set_xlabel("Frequency")
axes[2][1].set_xlabel("Frequency")

