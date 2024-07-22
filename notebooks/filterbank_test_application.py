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
os.chdir('/home/jasminekobayashi/gh_repos/idea-lab-sw-isax')
import fdl21.data.prototyping_metrics as pm

# %% Load test OMNI data
omni_path = '/mnt/c/sw-data/nasaomnireader/'
year = '2019'
month = '05'

cdf_file_path = omni_path + year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

mag_df = pm.read_OMNI_dataset(cdf_file_path)

# %% Select specific single day
day = 15
start = dt.datetime(year=int(year),month=int(month),day=day,hour=0)
end = dt.datetime(year=int(year),month=int(month),day=day+1,hour=0)

mag_df = mag_df[start:end]
# %%
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

# %%

