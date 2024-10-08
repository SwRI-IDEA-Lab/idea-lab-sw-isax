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
# %% Ch. 15 Formula
def moving_avg_freq_response(f,window=dt.timedelta(minutes=3000),cadence=dt.timedelta(minutes=1)):
    n = int(window.total_seconds()/cadence.total_seconds())
    numerator = np.sin(np.pi*f*window.total_seconds())
    denominator = window.total_seconds()*np.sin(np.pi*f)
    return abs(numerator/denominator)

# %%
# %% FT
cadence = dt.timedelta(seconds=60)

mag_df.sort_index(inplace=True)
mag_df.interpolate(method='index', kind='linear',limit_direction='both',inplace=True)
df_index=pd.date_range(start=mag_df.index[0], end=mag_df.index[-1], freq=cadence)

sig_fft_df = fft.fftn(mag_df - mag_df.mean(),axes=0)

# Smoothing filter
FB = fb.filterbank()
FB.build_triangle_fb(num_fft_bands=int(mag_df.shape[0])+1,sample_rate=1/60) # triangle filters are not necessary; I just wanted to make an fftfreq spectrum

window = dt.timedelta(seconds=18000)
smoothing_fr = moving_avg_freq_response(f=FB.fftfreq[1:],
                                        window=window,
                                        cadence=cadence)

# %%
plt.plot(FB.fftfreq[1:],smoothing_fr)
plt.title("Ch. 15 Formula Frequency Response (vs. Hz Frequency)")
plt.xlabel("Frequency (Hz)")

# %%
# %% plot frequency response
plt.plot(FB.fftfreq[1:],smoothing_fr)
plt.xlim(0,0.001)
plt.title("Ch. 15 Formula Frequency Response (vs. Hz Frequency)")
plt.xlabel("Frequency (Hz)")
plt.show()

# %%
# %% Apply moving average "filter"
Y = sig_fft_df.ravel()*smoothing_fr
filtered_y = np.real(fft.ifft(Y))

# %%
# %% smoothing in time chunking
smooth_y = tc.preprocess_smooth_detrend(mag_df=mag_df-mag_df.mean(),
                                        cols=cols,
                                        detrend_window=dt.timedelta(seconds=0),
                                        smooth_window=window)

# %%
# %% Compare
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,smooth_y,color='tab:green',linewidth=3,label='time_chunking smoothing')
plt.plot(mag_df.index,filtered_y,color='tab:orange',label='smoothing via Fourier')
plt.title(f'Smoothing window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %% [markdown]
# # Manual build - Moving Avg. filter

# %%
data_len = mag_df.shape[0]                              # length of data (i.e. number of samples)
sample_numbers = np.linspace(0,data_len,data_len)       # array of sample numbers
n = int(window.total_seconds()/cadence.total_seconds()) # number of "sample points" in the window (w.r.t. cadence)

# %%
# Make the box
box_filter = np.zeros(data_len) # start with array of zeros of size data_len
# print(box_filter.shape)

dx = (data_len-n)//2            # the number of zeros to center the box
box_filter[dx:dx+n] = 1         # set values to one
print(f'sum of ones (should equal {n}): {sum(box_filter)}')

# Normalize the box
box_filter = box_filter/np.sum(box_filter)

# visualize
plt.plot(sample_numbers,box_filter)

# %%
# FFT of the box filter
freq_resp = fft.fft(box_filter)

# %%
freq_resp.shape

# %%
plt.plot(np.abs(freq_resp))
plt.title('FFT of Box Filter')

# %%
plt.plot(np.abs(freq_resp))
plt.title('FFT of Box Filter')
plt.xlim(0,100)

# %%
plt.plot(sample_numbers/(1e5),np.abs(freq_resp))
plt.xlim(0,0.001)
plt.xlabel('sample number/(1e5)')

# %%
plt.plot(FB.fftfreq[1:],np.abs(freq_resp))
plt.xlim(0,0.001)
plt.xlabel('Frequency (Hz)')

# %%
plt.plot(FB.fftfreq[1:]*2,np.abs(freq_resp))
plt.xlim(0,0.001)
plt.xlabel('(Frequency (Hz)*2)')

# %%
# %% Apply moving average filter
Y_box = sig_fft_df.ravel()*abs(freq_resp)
box_filtered_y = np.real(fft.ifft(Y_box))

# %%
# %% Compare
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,smooth_y,color='tab:green',linewidth=3,label='time_chunking smoothing')
plt.plot(mag_df.index,filtered_y,color='tab:orange',label='Fourier: Ch.15 formula')
plt.plot(mag_df.index,box_filtered_y,color='tab:red',label='Fourier: manually-built box filter')
plt.title(f'Smoothing window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %%



