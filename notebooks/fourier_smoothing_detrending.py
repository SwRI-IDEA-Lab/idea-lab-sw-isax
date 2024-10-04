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

_FILE_DIR = os.path.dirname( os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_FILE_DIR)
sys.path.append(_SRC_DIR)

# local imports
import fdl21.data.prototyping_metrics as pm
import fdl21.utils.time_chunking as tc
import fdl21.data.build_filterbanks as fb

# %%
def moving_avg_freq_response(f,window=dt.timedelta(minutes=3000),cadence=dt.timedelta(minutes=1)):
    # TODO: Concern = does centered windows change the response?
    
    numerator = np.sin(np.pi*f*window.total_seconds())
    denominator = window.total_seconds()*np.sin(np.pi*f)
    return abs(numerator/denominator)

# %% Get data
year = '2019'
month = '05'
test_cdf_file_path =_SRC_DIR+fb._OMNI_MAG_DATA_DIR+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

mag_df = fb.get_test_data(fname_full_path=test_cdf_file_path)
mag_df

# %% FT
cadence = dt.timedelta(seconds=60)
cols = ['BY_GSE']

mag_df=mag_df[cols]
mag_df.sort_index(inplace=True)
mag_df.interpolate(method='index', kind='linear',limit_direction='both',inplace=True)
df_index=pd.date_range(start=mag_df.index[0], end=mag_df.index[-1], freq=cadence)

sig_fft_df = fft.fftn(mag_df - mag_df.mean(),axes=0)


# Smoothing filter
FB = fb.filterbank()
FB.build_triangle_fb(num_fft_bands=int(mag_df.shape[0])+1,sample_rate=1/60) # triangle filters are not necessary; I just wanted to make an fftfreq spectrum

window = dt.timedelta(seconds=18000)
smoothing_ft = moving_avg_freq_response(f=FB.fftfreq[1:],
                                        window=window,
                                        cadence=cadence)
# %% plot frequency response
plt.plot(FB.fftfreq[1:],smoothing_ft)
plt.show()

# %% Apply moving average "filter"
Y = sig_fft_df.ravel()*smoothing_ft
filtered_y = np.real(fft.ifft(Y))

# %% smoothing in time chunking
smooth_y = tc.preprocess_smooth_detrend(mag_df=mag_df-mag_df.mean(),
                                        cols=cols,
                                        detrend_window=dt.timedelta(seconds=0),
                                        smooth_window=window)

# %% Compare
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original')
plt.plot(mag_df.index,filtered_y,label='smoothing via Fourier')
plt.plot(mag_df.index,smooth_y,label='original smoothing')
plt.title(f'Fourier application of smoothing: window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %%
