import cdflib
import numpy as np

from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from multiprocessing import Pool
from scipy.stats import norm

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

import os
_FILE_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.dirname(_FILE_DIR)
os.chdir(_SRC_DIR)
import fdl21.data.prototyping_metrics as pm
import fdl21.utils.time_chunking as tc
# %% Load test OMNI data
_PSP_MAG_DATA_DIR = '/sw-data/psp/mag_rtn/'
_WIND_MAG_DATA_DIR = '/sw-data/wind/mfi_h2/'
_OMNI_MAG_DATA_DIR = '/sw-data/nasaomnireader/'
year = '2019'
month = '05'

cdf_file_path = _SRC_DIR+_OMNI_MAG_DATA_DIR + year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

mag_df = pm.read_OMNI_dataset(cdf_file_path)
mag_df.interpolate(inplace=True)

# %% Select specific single day
day = 15
start = dt.datetime(year=int(year),month=int(month),day=day,hour=0)
end = dt.datetime(year=int(year),month=int(month),day=day+1,hour=0)

mag_df = mag_df[start:end]
# %% Construct Mel Bank
f1, f2 = 0, 5
melmat, (melfreq, fftfreq) = melbank.compute_melmat(2, f1, f2, num_fft_bands=100000)

# Blue
DC_filter = 1-melmat[0,:]
minin = (DC_filter == np.min(DC_filter)).nonzero()[0][0]
DC_filter[minin:] = 0
melmat = np.append(DC_filter[None,:], melmat, axis=0)

# Red
HF_filter = 1-melmat[-1,:]
minin = (HF_filter == np.min(HF_filter)).nonzero()[0][0]
HF_filter[0:minin] = 0
melmat = np.append(melmat, HF_filter[None,:], axis=0)

# Plot filter banks & Mel matrix
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(fftfreq, melmat.T)
ax.grid(True)
ax.set_ylabel('Weight')
ax.set_xlabel('Frequency / Hz')
ax.set_xlim((f1, f2))
ax2 = ax.twiny()
ax2.xaxis.set_ticks_position('top')
ax2.set_xlim((f1, f2))
ax2.xaxis.set_ticks(melbank.mel_to_hertz(melfreq))
ax2.xaxis.set_ticklabels(['{:.0f}'.format(mf) for mf in melfreq])
ax2.set_xlabel('Frequency / mel')
plt.tight_layout()

fig, ax = plt.subplots()
ax.matshow(melmat)
plt.axis('equal')
plt.axis('tight')
plt.title('Mel Matrix')
plt.tight_layout()

# %% FFT
x = mag_df.index
y = mag_df['BY_GSE']

sig_fft = fft.fft(np.array(y)-np.mean(y))
sample_freq = fft.fftfreq(y.shape[0],d=1/150)
power = np.abs(sig_fft)**2
plt.plot(sample_freq,power)

# %% IFFT
high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > 100] = 0

filtered_sig = fft.ifft(high_freq_fft)
plt.plot(mag_df.index,y-np.mean(y),linewidth=5, label='Original signal')
plt.plot(mag_df.index,filtered_sig, label='Filtered signal')

# %% Filter bank application and decomposition
total = np.real(y.copy()*0)
total_paa = np.real(y.copy()*0)
fig = plt.figure(figsize=(24, 8))
gs = fig.add_gridspec(ncols = 3, nrows = 8, figure = fig ,wspace=0.2,hspace=0, )

for i in range(melmat.shape[0]):
    filter = np.interp(np.abs(sample_freq), fftfreq, melmat[i,:], left=None, right=None, period=None)
    filteredYF = sig_fft.copy()
    filteredYF = filteredYF*filter

    filtered_sig = np.real(fft.ifft(filteredYF))

    total = total + filtered_sig

    word_size = 20*(i + 1 + 3*np.max([0, i-2]) )
    paa = PiecewiseAggregateApproximation(word_size)
    paa_sequence = paa.fit_transform(filtered_sig[None,:]).squeeze()

    dx = (np.max(x)-np.min(x))/(word_size)
    # xpaa = np.linspace(np.min(x), np.max(x), word_size)#np.arange(np.min(x), np.max(x), dx)
    xpaa = np.min(x) + np.arange(0,word_size)/(word_size)*(np.max(x)-np.min(x))

    paa_sfull = total.copy()*0

    for j, segmentx in enumerate(xpaa):
        paa_sfull[np.array(x)>=segmentx] = paa_sequence[j]

    total_paa = total_paa + paa_sfull    

    ax0 = fig.add_subplot(gs[2*i:2*i+2,1])    
    ax0.plot(x, filtered_sig)
    ax0.plot(x, paa_sfull, c='r')
    # ax0.set_ylim([-3,3.5])
    # if i<3:
    ax0.set_xticks([])
    ax0.set_yticks([])

    if i==0:
        ax0.set_title('Filter bank decomposition')


ax0 = fig.add_subplot(gs[3:5,2])   
ax0.plot(x[0:-1:20], total_paa[0:-1:20], c='r')
ax0.set_title('Series recovered from filter bank PAA')
ax0.set_xticks([])
ax0.set_yticks([])


ax0 = fig.add_subplot(gs[3:5,0])   
ax0.plot(x, y-np.mean(y))
ax0.set_title('Original series')
# ax0.plot(x[0:-1:20], total_paa[0:-1:20], c='r')
ax0.set_xticks([])
ax0.set_yticks([])


ax = fig.add_subplot(gs[0:2,0])  
ax.plot(fftfreq, melmat.T)
ax.grid(True)
ax.set_ylabel('Weight')
ax.set_xlabel('Frequency / Hz')
ax.set_xlim((f1, f2))
ax.set_title('Mel filter bank')
ax.set_xticks([])
