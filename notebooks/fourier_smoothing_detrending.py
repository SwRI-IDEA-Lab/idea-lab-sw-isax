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

sig_fft_df = fft.fftn(mag_df - mag_df.mean(),axes=0)

# %% [markdown]
# # Smoothing via Convolution in the Time Domain
# The Smoothing (and Detrending) code from "time_chunking.py" (not excutable here)
# 
# ```python
# mag_df=mag_df[cols]
# mag_df.sort_index(inplace=True)
# preprocessed_mag_df = mag_df.copy()
# 
# if detrend_window > timedelta(seconds=0):
#     LOG.debug('Detrending')
#     smoothed = preprocessed_mag_df.rolling(detrend_window,
#         center=True
#     ).mean()
#     # Subtract the detrend_window (e.g. 30 minutes or 1800s) to detrend
#     preprocessed_mag_df = preprocessed_mag_df - smoothed
# 
# if smooth_window > timedelta(seconds=0):
#     LOG.debug('Smoothing')
#     preprocessed_mag_df = preprocessed_mag_df.rolling(smooth_window,
#         center=True
#     ).mean()
# ```

# %%
window = dt.timedelta(seconds=18000)

# %%
# %% Smoothing from time_chunking code script
smooth_y = tc.preprocess_smooth_detrend(mag_df=mag_df-mag_df.mean(),
                                        cols=cols,
                                        detrend_window=dt.timedelta(seconds=0),
                                        smooth_window=window)

# %%
# %% Compare
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,smooth_y,color='tab:green',linewidth=3,label='Convolution (time_chunking smoothing)')
plt.title(f'Smoothing window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

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
    numerator = np.sin(np.pi*f*window.total_seconds())
    denominator = window.total_seconds()*np.sin(np.pi*f)
    return abs(numerator/denominator)

# %%
# Build theoretical frequency response 
FB = fb.filterbank()
FB.build_triangle_fb(num_fft_bands=int(mag_df.shape[0])+1,sample_rate=1/60) # triangle filters are not necessary; I just wanted to make an fftfreq spectrum

FR_theory = moving_avg_freq_response(f=FB.fftfreq[1:],
                                        window=window,
                                        cadence=cadence)

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Theoretical Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(FB.fftfreq[1:],FR_theory)
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_title('Full Spectrum')
axes[1].plot(FB.fftfreq[1:],FR_theory)
axes[1].set_xlim(0,0.001)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_title('Zoomed in')

# %%
# Apply filter: theoretical frequency response
Y = sig_fft_df.ravel()*FR_theory
filtered_y = np.real(fft.ifft(Y))

# %%
# Compare
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,smooth_y,color='tab:green',linewidth=3,label='Convolution (time_chunking smoothing)')
plt.plot(mag_df.index,filtered_y,color='tab:orange',label='Fourier: Theory (Ch. 15 formula)')
plt.vlines(min(mag_df.index)+window,-100,100,linestyles='dashed',color='black')
plt.vlines(max(mag_df.index)-window,-100,100,linestyles='dashed',color='black')
plt.title(f'Smoothing window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %% [markdown]
# # Empirical application of moving avg. filter
# Manually build the rectangular moving average filter and take the FFT.
# 

# %%
# useful numbers
data_len = mag_df.shape[0]                              # length of data (i.e. number of samples)
sample_numbers = np.linspace(0,data_len,data_len)       # array of sample numbers
n = int(window.total_seconds()/cadence.total_seconds()) # number of "sample points" in the window (w.r.t. cadence)

# %%
# Make the box
box_filter = np.zeros(data_len) # start with array of zeros of size data_len
print("length of filter matches data length:",len(box_filter)==data_len)

dx = (data_len-n)//2            # the number of zeros to center the box

box_filter[dx:dx+n] = 1         # set values to one
print(f'sum of ones (should equal {n}): {sum(box_filter)}')

# Normalize the box
box_filter = box_filter/np.sum(box_filter)

# visualize
plt.plot(sample_numbers,box_filter)

# %%
# FFT of the box filter
FR_empirical = fft.fft(box_filter)
#print(FR_empirical.shape)

# %%
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
axes[0].plot(np.abs(FR_empirical))
axes[0].set_title('Full Spectrum')
axes[1].plot(np.abs(FR_empirical))
axes[1].set_xlim(0,100)
axes[1].set_title('Zoomed in')
fig.suptitle('Empirical Frequency Response (FFT of box filter)',fontsize=18)

# %%
# %% Apply empirical filter
Y_box = sig_fft_df.ravel()*abs(FR_empirical)
box_filtered_y = np.real(fft.ifft(Y_box))

# %%
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,smooth_y,color='tab:green',linewidth=3,label='Convolution (time_chunking smoothing)')
plt.plot(mag_df.index,filtered_y,color='tab:orange',label='Fourier: Theory (Ch. 15 formula)')
plt.plot(mag_df.index,box_filtered_y,color='tab:red',label='Fourier: Empirical (FFT of built filter)')
plt.vlines(min(mag_df.index)+window,-100,100,linestyles='dashed',color='black')
plt.vlines(max(mag_df.index)-window,-100,100,linestyles='dashed',color='black')
plt.title(f'Smoothing window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %% [markdown]
# Goal: we want red (empirical FR) and orange (theoretical FR) line to match green line (moving average filter (smoothing)).

# %% [markdown]
# -------

# %% [markdown]
# # Let the investigation begin!

# %% [markdown]
# ## Symmetric vs. not-symmetric
# The empirical frequency response has a kinda U-shape, while the formula from theory (essentially) diminishes forever. Let's see if excluding higher frequencies makes a difference.

# %%
FR_empirical_no_HF = FR_empirical.copy()
FR_empirical_no_HF[1200:] = 0

# %%
Y_box2 = sig_fft_df.ravel()*abs(FR_empirical_no_HF)
box_filtered_y2 = np.real(fft.ifft(Y_box2))

# %%
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,smooth_y,color='tab:green',linewidth=3,label='time_chunking smoothing')
plt.plot(mag_df.index,filtered_y,color='tab:orange',linewidth=2,label='Fourier: Theory (Ch.15 formula)')
plt.plot(mag_df.index,box_filtered_y,color='tab:red',label='Fourier: Empirical (FFT of built filter)')
plt.plot(mag_df.index,box_filtered_y2,color='cyan',linestyle='dashed',label='Fourier: Empirical (FFT of built filter) \n[no HF]')
plt.vlines(min(mag_df.index)+window,-100,100,linestyles='dashed',color='black')
plt.vlines(max(mag_df.index)-window,-100,100,linestyles='dashed',color='black')
plt.title(f'Smoothing window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %% [markdown]
# **Section Conclusion:** The higher frequencies involved in the empirical frequency response, make a difference.

# %% [markdown]
# ## Compare Theoretical vs. Empirical Frequency Response

# %%
plt.plot(FR_theory,label='theory')
plt.plot(abs(FR_empirical),label='empirical')
plt.title('Frequency Responses (by index)')
plt.legend()
plt.xlim(0,100)

# %% [markdown]
# ---
# 
# ## Finding the factor

# %%
plt.plot(FR_theory,label='theory')
plt.plot(sample_numbers*2,abs(FR_empirical),label='empirical')
plt.title('Frequency Responses (factor = 2?)')
plt.legend()
plt.xlim(0,100)

# %%
plt.plot(FR_theory,label='theory')
plt.plot(sample_numbers*np.pi,abs(FR_empirical),label='empirical')
plt.title('Frequency Responses (factor = $\pi$?)')
plt.legend()
plt.xlim(0,100)

# %% [markdown]
# **Section conclusion:** It appears the theoretical and empirical frequency responses differ by a factor of ~2 (more so than $\pi$)
# 
# ***Side Note:*** I remember seeing something about 1/2 in relation to something called Nyquist frequency and the sampling theorem.

# %% [markdown]
# ## RFFT vs. FFT

# %%
RFR_empirical = fft.rfft(box_filter)

# %%
plt.plot(abs(RFR_empirical))

# %%
plt.plot(abs(FR_empirical),color='tab:orange',label='empirical ((complex) FFT)')
plt.plot(abs(RFR_empirical),color='tab:green',linestyle='dashed',label='empirical (real FFT)')
plt.title('Frequency Responses (real FFT)')
plt.legend()

# %%
plt.plot(FR_theory,label='theory')
plt.plot(abs(FR_empirical),label='empirical ((complex) FFT)')
plt.plot(abs(RFR_empirical),linestyle='dashed',label='empirical (real FFT)')
plt.title('Frequency Responses (complex vs. real FFT)')
plt.legend()
plt.xlim(0,100)

# %% [markdown]
# ---

# %% [markdown]
# # Supposed Solution
# 
# 1. Make sure using appropriate range of frequencies for the theoretical frequency response formula 
#     * (i.e. make sure appropriately implementing the sampling theorem and not using frequencies above the Nyquist frequency)
# 2. Use the "one-sided" FFT instead of the "two-sided" FFT (which includes complex values) 
#     * i.e. Use Scipy's `rfft` (requires real value input) instead of `fft`

# %%
sig_rfft_df = fft.rfftn(mag_df - mag_df.mean(),axes=0)

# %% [markdown]
# ## Use appropriate range of frequencies for the theoretical frequency response

# %%
# %% Ch. 15 Formula
def moving_avg_freq_response(f,window=dt.timedelta(minutes=3000),cadence=dt.timedelta(minutes=1)):
    n = int(window.total_seconds()/cadence.total_seconds())
    numerator = np.sin(np.pi*f*n)
    denominator = n*np.sin(np.pi*f)
    return abs(numerator/denominator)

# %% [markdown]
# **NOTE:** This version utilizes the `n = int(window.total_seconds()/cadence.total_seconds())` instead of `window.total_seconds()` in the first version

# %%
# Build theoretical frequency response 
data_len = mag_df.shape[0]  
freq_spectrum = np.linspace(0.0001,data_len/2,(data_len//2)+1)
FR_theory_r = moving_avg_freq_response(f=freq_spectrum,
                                        window=window,
                                        cadence=cadence)

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Theoretical Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum,FR_theory_r)
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum,FR_theory_r)
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')

# %% [markdown]
# **NOTE:** the frequencies now range from 0 to 100. But this is NOT in hertz.

# %% [markdown]
# ## DFT frequency to hertz
# (The following details is from [this video](https://youtu.be/QmgJmh2I3Fw?feature=shared&t=795))
# 
# Let's denote the frequency that comes out doing a DFT to a signal of $N$ samples as $k$.
# 
# This frequency $k$ is equal to the number of cycles ($\alpha$) completed within a time period, $T$, which is equal to the length of the time domain signal.
# 
# $$k = \frac{\alpha}{T}$$
# 
# To convert $k$ into a frequency in hertz.
# 
# \begin{equation}
# f_{\text{hz}} = \frac{k}{\text{length of time signal (in seconds)}} = \frac{k \cdot f_s}{N}
# \end{equation}
# 
# (where $f_s$ is the sampling rate)

# %%
freq_spectrum_hz = freq_spectrum/(data_len*cadence.total_seconds())

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Theoretical Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum_hz,FR_theory_r)
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum_hz,FR_theory_r)
axes[1].set_xlim(0,0.001)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_title('Zoomed in')

# %% [markdown]
# (This is more similar to what we got in the first version.)

# %%
# Apply filter: theoretical frequency response
Y_r = sig_rfft_df.ravel()*FR_theory_r
filtered_y_r = np.real(fft.irfft(Y_r))

# %%
# Compare
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,smooth_y,color='tab:green',linewidth=3,label='Convolution (time_chunking smoothing)')
plt.plot(mag_df.index[1:],filtered_y_r,color='tab:orange',label='Fourier: Theory (Ch. 15 formula)')
plt.vlines(min(mag_df.index)+window,-100,100,linestyles='dashed',color='black')
plt.vlines(max(mag_df.index)-window,-100,100,linestyles='dashed',color='black')
plt.title(f'Smoothing window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %% [markdown]
# ## Use `rfft` for the empirical frequency response

# %%
# FFT of the box filter
RFR_empirical = fft.rfft(box_filter)
#print(RFR_empirical.shape)

# %%
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
axes[0].plot(np.abs(RFR_empirical))
axes[0].set_title('Full Spectrum')
axes[1].plot(np.abs(RFR_empirical))
axes[1].set_xlim(0,100)
axes[1].set_title('Zoomed in')
fig.suptitle('Empirical Frequency Response (FFT of box filter)',fontsize=18)

# %% [markdown]
# **Notice:** We no longer have that U-shape in the frequency response. 
# 
# (Which turned out to be the "two-sided" frequency response, where the absolute value of the "positive" and "negative" frequencies mirrored each other across the Nyquist frequency.)

# %%
# %% Apply empirical filter
Y_box_r = sig_rfft_df.ravel()*abs(RFR_empirical)
box_filtered_y_r = np.real(fft.irfft(Y_box_r))

# %%
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,smooth_y,color='tab:green',linewidth=5,label='Convolution (time_chunking smoothing)')
plt.plot(mag_df.index[1:],filtered_y_r,color='tab:orange',linewidth=3,label='Fourier: Theory (Ch. 15 formula)')
plt.plot(mag_df.index[1:],box_filtered_y_r,color='black',linestyle='dashdot',label='Fourier: Empirical (FFT of built filter)')
plt.vlines(min(mag_df.index)+window,-100,100,linestyles='dashed',color='gray')
plt.vlines(max(mag_df.index)-window,-100,100,linestyles='dashed',color='gray')
plt.title(f'Smoothing window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %% [markdown]
# Huzzah!
# 
# Aside from the slight aliasing at the edges (due to the assumption of periodicity from the DFT), all methods of smoothing match up!
# 
# (Supposedly zero-padding the time domain signal may help with the aliasing problem at the edges, but we can look into that later.)


