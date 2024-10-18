# %%
# %% libraries
import pandas as pd
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt

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
# # Detrending & Smoothing via Convolution in the Time Domain
# The Smoothing and Detrending code from "time_chunking.py" (not excutable here)
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
SM_window = dt.timedelta(seconds=3000)
DT_window = dt.timedelta(seconds=18000)

# %%
# %% Detrending from time_chunking code script

detrend_y = tc.preprocess_smooth_detrend(mag_df=mag_df-mag_df.mean(),
                                        cols=cols,
                                        detrend_window=DT_window,
                                        smooth_window=SM_window)

# %%
# %% Compare

plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,detrend_y,color='tab:green',linewidth=3,label='Convolution (time_chunking detrending)')
plt.title(f'SM window = {int(SM_window.total_seconds())}, DT window = {int(DT_window.total_seconds())} (sec)')
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
    numerator = np.sin(np.pi*f*n)
    denominator = n*np.sin(np.pi*f)
    return abs(numerator/denominator)

# %% [markdown]
# ## Detrending in the frequency domain
# 
# Since detrending in the time domain is `detrended_sig = signal - mvg_avg(window)`, the *frequency response* of detrending ($DT$) should be
# 
# \begin{equation}
# DT[f] = 1 - H[f] = 1 - \frac{\sin(\pi f M)}{M \sin(\pi f)}
# \end{equation}

# %%
# Build theoretical frequency response 
data_len = mag_df.shape[0]  
freq_spectrum = np.linspace(0.0001,data_len/2,(data_len//2)+1)
DT_theory = moving_avg_freq_response(f=freq_spectrum,
                                        window=DT_window,
                                        cadence=cadence)
DT_theory = 1 - DT_theory

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Theoretical Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum,DT_theory)
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum,DT_theory)
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')

# %% [markdown]
# Smoothing only uses the moving average frequency response formula directly
# 

# %%
SM_theory = moving_avg_freq_response(f=freq_spectrum,
                                        window=SM_window,
                                        cadence=cadence)

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Theoretical Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum,SM_theory)
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum,SM_theory)
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Detrending Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum,SM_theory,label='Smoothing')
axes[0].plot(freq_spectrum,DT_theory,label='Detrending')
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum,SM_theory,label='Smoothing')
axes[1].plot(freq_spectrum,DT_theory,label='Detrending')
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')
axes[0].legend()
axes[1].legend()

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Detrending Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum,SM_theory,linestyle='dotted',label='Smoothing')
axes[0].plot(freq_spectrum,DT_theory,linestyle='dotted',label='Detrending')
axes[0].plot(freq_spectrum,DT_theory*SM_theory,label='Detrend*Smooth')
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum,SM_theory,linestyle='dotted',label='Smoothing')
axes[1].plot(freq_spectrum,DT_theory,linestyle='dotted',label='Detrending')
axes[1].plot(freq_spectrum,DT_theory*SM_theory,label='Detrend*Smooth')
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')
axes[0].legend()
axes[1].legend()

# %% [markdown]
# In the time_chunking code, it appears we smooth a detrended signal (i.e. perform detrending before applying smoothing)

# %%
# Apply filter: theoretical frequency response
Y = (sig_fft_df.ravel()*DT_theory)*SM_theory
filtered_y = np.real(fft.irfft(Y))

# %%
# Compare
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,detrend_y,color='tab:green',linewidth=3,label='Convolution (time_chunking detrending)')
plt.plot(mag_df.index[1:],filtered_y,color='tab:orange',label='Fourier: Theory (Ch. 15 formula)')
# plt.vlines(min(mag_df.index)+(DT_window-SM_window),-100,100,linestyles='dashed',color='black')
# plt.vlines(max(mag_df.index)-(DT_window-SM_window),-100,100,linestyles='dashed',color='black')
# plt.title(f'Detrending window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()

# %%
FR_theory = DT_theory*SM_theory

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle("Theoretical Frequency Response (Ch. 15 formula)",fontsize=18)
axes[0].plot(freq_spectrum,FR_theory)
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_spectrum,FR_theory)
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')

# %%
# Apply filter: theoretical frequency response
Y = sig_fft_df.ravel()*FR_theory
filtered_y = np.real(fft.irfft(Y))

# %%
# Compare
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,detrend_y,color='tab:green',linewidth=3,label='Convolution (time_chunking detrending)')
plt.plot(mag_df.index[1:],filtered_y,color='tab:orange',label='Fourier: Theory (Ch. 15 formula)')
# plt.vlines(min(mag_df.index)+(DT_window-SM_window),-100,100,linestyles='dashed',color='black')
# plt.vlines(max(mag_df.index)-(DT_window-SM_window),-100,100,linestyles='dashed',color='black')
# plt.title(f'Detrending window = {int(window.total_seconds())} seconds')
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
n_sm= int(SM_window.total_seconds()/cadence.total_seconds()) # number of "sample points" in the window (w.r.t. cadence)
n_dt = int(DT_window.total_seconds()/cadence.total_seconds())

# %%
# Make the box
box_filter_SM = np.zeros(data_len) # start with array of zeros of size data_len
print("length of filter matches data length:",len(box_filter_SM)==data_len)

dx = (data_len-n_sm)//2            # the number of zeros to center the box

box_filter_SM[dx:dx+n_sm] = 1         # set values to one
print(f'sum of ones (should equal {n_sm}): {sum(box_filter_SM)}')

# Normalize the box
box_filter_SM = box_filter_SM/np.sum(box_filter_SM)


# %%
# Make the box
box_filter_DT = np.zeros(data_len) # start with array of zeros of size data_len
print("length of filter matches data length:",len(box_filter_DT)==data_len)

dx = (data_len-n_dt)//2            # the number of zeros to center the box

box_filter_DT[dx:dx+n_dt] = 1         # set values to one
print(f'sum of ones (should equal {n_dt}): {sum(box_filter_DT)}')

# Normalize the box
box_filter_DT = box_filter_DT/np.sum(box_filter_DT)


# %%
# visualize
plt.plot(sample_numbers,box_filter_SM,label='smoothing box')
plt.plot(sample_numbers,box_filter_DT,label='detrending')
plt.legend()

# %%
# FFT of the box filter
SM_empirical = fft.rfft(box_filter_SM)
#print(FR_empirical.shape)

# %%
# FFT of the box filter
DT_empirical = fft.rfft(box_filter_DT)
DT_empirical = 1 - abs(DT_empirical)
#print(FR_empirical.shape)

# %%
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
axes[0].plot(np.abs(SM_empirical))
axes[0].plot(SM_theory,linestyle='dashed')
axes[0].set_title('Full Spectrum')
axes[1].plot(np.abs(SM_empirical))
axes[1].set_xlim(0,100)
axes[1].set_title('Zoomed in')
fig.suptitle('Empirical Frequency Response (FFT of box filter)',fontsize=18)

# %%
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
axes[0].plot(np.abs(DT_empirical))
axes[0].plot(DT_theory,linestyle='dashed')
axes[0].set_title('Full Spectrum')
axes[1].plot(np.abs(DT_empirical))
axes[1].set_xlim(0,100)
axes[1].set_title('Zoomed in')
fig.suptitle('Empirical Frequency Response (FFT of box filter)',fontsize=18)

# %%
FR_empirical = DT_empirical*SM_empirical

# %%
# Apply empirical filter
Y_box = sig_fft_df.ravel()*abs(FR_empirical)
box_filtered_y = np.real(fft.irfft(Y_box))

# %%
plt.plot(mag_df.index,mag_df[:]-mag_df.mean(),label='original data')
plt.plot(mag_df.index,detrend_y,color='tab:green',linewidth=5,label='Convolution (time_chunking detrending)')
plt.plot(mag_df.index[1:],filtered_y,color='tab:orange',linewidth=3,label='Fourier: Theory (Ch. 15 formula)')
plt.plot(mag_df.index[1:],box_filtered_y,color='black',linestyle='dotted',label='Fourier: Empirical (FFT of built filter)')
# plt.vlines(min(mag_df.index)+window,-100,100,linestyles='dashed',color='gray')
# plt.vlines(max(mag_df.index)-window,-100,100,linestyles='dashed',color='gray')
# plt.title(f'Detrending window = {int(window.total_seconds())} seconds')
plt.legend()
plt.show()


