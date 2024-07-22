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

import os
os.chdir('/home/jasminekobayashi/gh_repos/idea-lab-sw-isax')
import fdl21.data.prototyping_metrics as pm

# %%
omni_path = '/mnt/c/sw-data/nasaomnireader/'
year = '2019'
month = '05'

cdf_file_path = omni_path + year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

mag_df = pm.read_OMNI_dataset(cdf_file_path)

