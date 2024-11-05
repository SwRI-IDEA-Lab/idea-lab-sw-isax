# %%
# %% libraries
import pandas as pd
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt

from scipy import fft
from sklearn.metrics import r2_score

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

# %% [markdown]
# # Summary of theoretical formulas
# 
# **(Formula from [Ch. 15 of *Digital Signal Processing Textbook*](https://www.dspguide.com/CH15.PDF)**)
# 
# Frequency response of an $M$ point moving average filter. The frequency, $f$, runs between $0$ and $0.5$. For $f = 0$, use $H[f] = 1$
# 
# $$H[f] = \frac{\sin(\pi f M)}{M\sin(\pi f)}$$
# 
# **Frequency response of detrending (DT)** with window $W_d$,
# 
# \begin{align*}
# 
# \widetilde{DT}[f] &= 1 - H_d[f] \\
# 
# &= 1 - \frac{\sin (\pi f W_d)}{W_d \sin (\pi f)}
# 
# \end{align*}
# 
# **Frequency response of smoothing (SM)** with window $W_s$,
# 
# \begin{align*}
# 
# \widetilde{SM}[f] &= H_s[f] \\
# 
# &= \frac{\sin (\pi f W_s)}{W_s \sin (\pi f)}
# 
# \end{align*}
# 
# **Frequency response of *both* detrending and smoothing** (just product of the above two)
# 
# \begin{align*}
# 
# \widetilde{DTSM}[f] &=  \widetilde{DT}[f] \cdot \widetilde{SM}[f] \\
# 
# &= (1-H_d[f]) \cdot (H_s[f]) \\
# 
# &= \left(1 - \frac{\sin (\pi f W_d)}{W_d \sin (\pi f)}\right) \cdot \left(\frac{\sin (\pi f W_s)}{W_s \sin (\pi f)}\right) 
# 
# \end{align*}

# %% [markdown]
# # Create Filterbanks

# %%
# Moving average filterbanks
DTSM = fb.filterbank(data_len=len(mag_df),
                    cadence=dt.timedelta(seconds=60))
DTSM.build_DTSM_fb(windows=[2000,6000,18000,54000])
DTSM.add_mvgavg_DC_HF()
fb.visualize_filterbank(fb_matrix=DTSM.fb_matrix,
                        fftfreq=DTSM.freq_hz_spec,
                        xlim=(0,0.002))

# %%
# triangle filterbanks
tri = fb.filterbank(data_len=len(mag_df),
                    cadence=dt.timedelta(seconds=60))
tri.build_triangle_fb((0.0,np.sort(DTSM.center_freq)[-1]),
                      center_freq=np.sort(DTSM.center_freq[1:-1]))
tri.add_DC_HF_filters()
fb.visualize_filterbank(fb_matrix=tri.fb_matrix,
                        fftfreq=tri.freq_hz_spec,
                        xlim=(0.0,0.0015))

# %%
# plot all
plt.figure(figsize=(10,5))
for m_bank in DTSM.fb_matrix:
    plt.plot(DTSM.freq_hz_spec,m_bank,linewidth=2)
for t_bank in tri.fb_matrix:
    plt.plot(tri.freq_hz_spec,t_bank,linestyle='dashed')

plt.xlim(0.0,0.0015)
plt.xlabel('Frequency (Hz)')
plt.grid()
plt.title('Both filterbank types in single plot')
plt.show()

# %%
# Sum of filterbank amplitudes
plt.plot(DTSM.freq_hz_spec,np.sum(abs(DTSM.fb_matrix),axis=0),label='$\sum$ DTSM filters')
plt.plot(tri.freq_hz_spec,np.sum(tri.fb_matrix,axis=0),label='$\sum$ triangles filters')
plt.xlabel('Frequency (hz)')
plt.title('Sum of filter amplitudes across all frequencies')
plt.legend()
plt.show()

# %%
# Plot frequency space totals (stem plot)
plt.stem(DTSM.freq_hz_spec,np.sum(abs(DTSM.fb_matrix),axis=0)-1,markerfmt='.')
plt.xlabel('Frequency (hz)')
plt.title('Sum of amplitudes (cont.): difference from 1')
plt.show()

# %%
DTSM_filtered, DTSM_paa = fb.visualize_filterbank_application(data_df=mag_df,
                                                            fb_matrix=DTSM.fb_matrix,
                                                            fftfreq=DTSM.freq_hz_spec,
                                                            data_col='BY_GSE',
                                                            cadence=dt.timedelta(minutes=1),
                                                            wordsize_factor = 3,
                                                            xlim = (0,DTSM.center_freq[-1]),
                                                            center_freq = DTSM.center_freq,
                                                            DC=DTSM.DC,
                                                            HF=DTSM.HF,
                                                            save_results=True)

# %%
tri_filtered, tri_paa = fb.visualize_filterbank_application(data_df=mag_df,
                                                            fb_matrix=tri.fb_matrix,
                                                            fftfreq=tri.freq_hz_spec,
                                                            data_col='BY_GSE',
                                                            cadence=dt.timedelta(minutes=1),
                                                            wordsize_factor = 3,
                                                            xlim = (0,tri.center_freq[-1]),
                                                            center_freq = tri.center_freq,
                                                            DC=tri.DC,
                                                            HF=tri.HF,
                                                            save_results=True)
# %%
# "filterbank" of Smoothing & detrending via convolution
convolution_filtered = np.zeros(DTSM_filtered.shape)
DTSM.windows.sort(reverse=True)
for i,w in enumerate(DTSM.windows[:-1]):
    filtered = tc.preprocess_smooth_detrend(mag_df=mag_df-mag_df.mean(),
                                            cols=cols,
                                            detrend_window=dt.timedelta(seconds=w),
                                            smooth_window=dt.timedelta(seconds=DTSM.windows[i+1]))
    convolution_filtered[i+1] = np.array(filtered[cols]).ravel()
# DC
DC_filtered = tc.preprocess_smooth_detrend(mag_df=mag_df-mag_df.mean(),
                                           cols=cols,
                                           detrend_window=dt.timedelta(seconds=0),
                                           smooth_window=dt.timedelta(seconds=max(DTSM.windows)))
convolution_filtered[0] = np.array(DC_filtered).ravel()
# HF
HF_filtered = tc.preprocess_smooth_detrend(mag_df=mag_df-mag_df.mean(),
                                           cols=cols,
                                           detrend_window=dt.timedelta(seconds=min(DTSM.windows)),
                                           smooth_window=dt.timedelta(seconds=0))
convolution_filtered[-1] = np.array(HF_filtered).ravel()

# %%
sum_DTSM_filtered = np.sum(DTSM_filtered,axis=0)
sum_tri_filtered = np.sum(tri_filtered,axis=0)
sum_conv_filtered = np.sum(convolution_filtered,axis=0)
real = real = np.array(mag_df-mag_df.mean()).ravel()

DTSM_r2 = r2_score(real,sum_DTSM_filtered)
tri_r2 = r2_score(real,sum_tri_filtered)
conv_r2 = r2_score(real,sum_conv_filtered)
 # %%
 # Plot Original vs. DTSM vs. Triangles 
plt.plot(mag_df.index,real,label='original')
plt.plot(mag_df.index,sum_DTSM_filtered,label=f'$\sum$ Moving Averages($R^2$: {DTSM_r2:.2f})')
plt.plot(mag_df.index,sum_tri_filtered,'--',label=f'$\sum$ triangle filterbanks ($R^2$: {tri_r2:.2f})')
plt.legend()
plt.show()

# %%
# Plot Original vs. Convolution vs. Triangles
plt.figure(figsize=(10,5))
plt.plot(mag_df.index,real,label='original')
plt.plot(mag_df.index,sum_conv_filtered,linestyle='dashed',label=f'$\sum$ convolution filtered ($R^2$: {conv_r2:.2f})')
# plt.plot(mag_df.index,sum_DTSM_filtered,label=f'$\sum$ Moving Averages($R^2$:{DTSM_r2:.2e})')
plt.plot(mag_df.index,sum_tri_filtered,linestyle='dashdot',label=f'$\sum$ triangle filterbanks ($R^2$: {tri_r2:.2f})')
plt.legend()
plt.show()

# %%
# residuals
DTSM_residual = real-sum_DTSM_filtered
tri_residual = real-sum_tri_filtered
conv_residual = real - sum_conv_filtered

residuals = {'DTSM':DTSM_residual,
            'Triangles':tri_residual,
            'Convolution':conv_residual}
colors = ['b','m','g']
lcolors = ['y','c','']


# %%
# Plot residuals: all in single plot
for i,selection in enumerate(['DTSM','Convolution','Triangles']):
    plt.stem(mag_df.index,residuals[selection],markerfmt=
    f'{colors[i]}.',linefmt=f'{lcolors[i]}-',label=f'{selection} (total: {sum(abs(residuals[selection])):.2e})')
    plt.legend()
plt.show()
# %%
# Plot residuals: individually
for i,selection in enumerate(['DTSM','Convolution','Triangles']):
    plt.stem(mag_df.index,residuals[selection],markerfmt=
    f'{colors[i]}.',linefmt=f'{lcolors[i]}',label=f'{selection}')
    plt.legend()
    plt.show()

# %%
