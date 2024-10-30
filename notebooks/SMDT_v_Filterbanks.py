# %%
# %% libraries
import pandas as pd
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt

from scipy import fft

import datetime as dt
import os,sys

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.dirname(_FILE_DIR)
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_MODEL_DIR)
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
test_cdf_file_path =_MODEL_DIR+fb._OMNI_MAG_DATA_DIR+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

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

# %%
data_len = mag_df.shape[0]  
sample_rate = 1/cadence.total_seconds()
freq_spectrum = np.linspace(0.0001,data_len/2,(data_len//2)+1)
freq_hz = freq_spectrum*sample_rate/(data_len)

# %% [markdown]
# # Windows1

# %%
window1_SM = dt.timedelta(seconds=500)
window1_DT = dt.timedelta(seconds=3000)

# %%
# Build theoretical frequency response 
FR1_smooth = moving_avg_freq_response(f=freq_spectrum,
                                        window=window1_SM,
                                        cadence=cadence)

# %%
FR1_theory = moving_avg_freq_response(f=freq_spectrum,
                                        window=window1_DT,
                                        cadence=cadence)
FR1_detrend = 1 - FR1_theory

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle(f"Windows1 (SM:{int(window1_SM.total_seconds())}; DT: {int(window1_DT.total_seconds())})",fontsize=18)
axes[0].plot(freq_hz,FR1_smooth,linestyle='dotted',label='Smoothing')
axes[0].plot(freq_hz,FR1_detrend,linestyle='dotted',label='Detrending')
axes[0].plot(freq_hz,FR1_detrend*FR1_smooth,linestyle='dashed',label='Detrend*Smooth')
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_hz,FR1_smooth,linestyle='dotted',label='Smoothing')
axes[1].plot(freq_hz,FR1_detrend,linestyle='dotted',label='Detrending')
axes[1].plot(freq_hz,FR1_detrend*FR1_smooth,label='Detrend*Smooth')
axes[1].set_xlim(0,0.004)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')
axes[0].legend()
axes[1].legend()
plt.show()

# %% [markdown]
# # Windows2

# %%
window2_SM = dt.timedelta(seconds=3000)
window2_DT = dt.timedelta(seconds=18000)

# %%
# Build theoretical frequency response 
FR2_smooth = moving_avg_freq_response(f=freq_spectrum,
                                        window=window2_SM,
                                        cadence=cadence)

# %%
FR2_theory = moving_avg_freq_response(f=freq_spectrum,
                                        window=window2_DT,
                                        cadence=cadence)
FR2_detrend = 1 - FR2_theory

# %%
# plot frequency response
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
fig.suptitle(f"Windows2 (SM:{int(window2_SM.total_seconds())}; DT: {int(window2_DT.total_seconds())})",fontsize=18)
axes[0].plot(freq_hz,FR2_smooth,linestyle='dotted',label='Smoothing')
axes[0].plot(freq_hz,FR2_detrend,linestyle='dotted',label='Detrending')
axes[0].plot(freq_hz,FR2_detrend*FR2_smooth,linestyle='dashed',label='Detrend*Smooth')
axes[0].set_xlabel("Frequency")
axes[0].set_title('Full Spectrum')
axes[1].plot(freq_hz,FR2_smooth,linestyle='dotted',label='Smoothing')
axes[1].plot(freq_hz,FR2_detrend,linestyle='dotted',label='Detrending')
axes[1].plot(freq_hz,FR2_detrend*FR2_smooth,label='Detrend*Smooth')
axes[1].set_xlim(0,0.004)
axes[1].set_xlabel("Frequency")
axes[1].set_title('Zoomed in')
axes[0].legend()
axes[1].legend()
plt.show()

# %% [markdown]
# # Comprehensive

# %%
# fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
# # fig.suptitle(f"Windows1 (SM:{int(window2_SM.total_seconds())}; DT: {int(window2_DT.total_seconds())})",fontsize=18)
# # axes[0].plot(freq_spectrum,FR1_smooth,linestyle='dotted',label='Smoothing')
# # axes[0].plot(freq_spectrum,FR1_detrend,linestyle='dotted',label='Detrending')
# axes[0].plot(freq_hz,FR1_detrend*FR1_smooth,label='window1')
# axes[0].plot(freq_hz,FR2_detrend*FR2_smooth,label='window2')
# axes[0].set_xlabel("Frequency")
# axes[0].set_title('Full Spectrum')
# # axes[1].plot(freq_spectrum,FR1_smooth,linestyle='dotted',label='Smoothing')
# # axes[1].plot(freq_spectrum,FR1_detrend,linestyle='dotted',label='Detrending')
# axes[1].plot(freq_hz,FR1_detrend*FR1_smooth,label=f'SM:{int(window1_SM.total_seconds())}; DT:{int(window1_DT.total_seconds())}')
# axes[1].plot(freq_hz,FR2_detrend*FR2_smooth,label=f'SM:{int(window2_SM.total_seconds())}; DT:{int(window2_DT.total_seconds())}')
# axes[1].set_xlim(0,0.004)
# axes[1].set_xlabel("Frequency")
# axes[1].set_title('Zoomed in')
# axes[0].legend()
# axes[1].legend()
# plt.show()

# %%
plt.plot(freq_hz,FR1_detrend*FR1_smooth,label=f'SM1:{int(window1_SM.total_seconds())}; DT1:{int(window1_DT.total_seconds())}')
plt.plot(freq_hz,FR2_detrend*FR2_smooth,label=f'SM2:{int(window2_SM.total_seconds())}; DT2:{int(window2_DT.total_seconds())}')
plt.xlabel("Frequency (Hz)")
plt.title('Smoothing & Detrending: Multiple windows')
plt.legend()
plt.grid()
plt.show()

# %%
FR1 = FR1_detrend*FR1_smooth
FR2 = FR2_detrend*FR2_smooth
cntr_freq1 = freq_hz[np.argmax(FR1)]
cntr_freq2 = freq_hz[np.argmax(FR2)]

# %%
test = fb.filterbank()
test.build_triangle_fb(num_bands=2,
                        frequencies=[0.0,cntr_freq2,cntr_freq1,freq_hz[180]],
                        num_fft_bands=int(1e6),
                        sample_rate=sample_rate)
# fb.visualize_filterbank(fb_matrix=test.fb_matrix,
#                         fftfreq=test.fftfreq,
#                         xlim=(0,0.01))

plt.plot(test.fftfreq,test.fb_matrix[0],label=f'filterbank[0] (cnt_freq: {cntr_freq2:.1e} hz)')
plt.plot(test.fftfreq,test.fb_matrix[1],label=f'filterbank[1] (cnt_freq: {cntr_freq1:.1e} hz)')
plt.legend()
plt.title("Build Similar Filterbank")
plt.xlabel("Frequency (Hz)")
plt.grid()
plt.show()

# %%
plt.plot(freq_hz,FR1_detrend*FR1_smooth,label=f'SM1:{int(window1_SM.total_seconds())}; DT1:{int(window1_DT.total_seconds())}')
plt.plot(freq_hz,FR2_detrend*FR2_smooth,label=f'SM2:{int(window2_SM.total_seconds())}; DT2:{int(window2_DT.total_seconds())}')
plt.plot(test.fftfreq,test.fb_matrix[0],linestyle='dashed',label=f'filterbank[0] (cnt_freq: {cntr_freq2:.1e} hz)')
plt.plot(test.fftfreq,test.fb_matrix[1],linestyle='dashed',label=f'filterbank[1] (cnt_freq: {cntr_freq1:.1e} hz)')
plt.legend()
plt.title('Compare SMDT windows with Filterbanks')
plt.xlabel('Frequency (Hz)')
plt.xlim(0,0.004)
plt.grid()
plt.show()
# %%
