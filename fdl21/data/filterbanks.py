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
import os,sys

import fdl21.data.prototyping_metrics as pm
import fdl21.utils.time_chunking as tc

omni_path = '/mnt/c/sw-data/nasaomnireader/'
year = '2019'
month = '05'


test_cdf_file_path = omni_path + year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'


_MODEL_DIR = os.path.dirname( os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_SRC_DIR)

_PSP_MAG_DATA_DIR = '/sw-data/psp/mag_rtn/'
_WIND_MAG_DATA_DIR = '/sw-data/wind/mfi_h2/'
_OMNI_MAG_DATA_DIR = '/sw-data/nasaomnireader/'
_SRC_DATA_DIR = os.path.join(_SRC_DIR,'data',)

def get_test_data(cdf_file_path,
                  start_date = dt.datetime(year=2019,month=5,day=15,hour=0),
                  end_date = dt.datetime(year=2019,month=5,day=16,hour=0)):
    mag_df = pm.read_OMNI_dataset(cdf_file_path)
    mag_df.interpolate(inplace=True)

    mag_df = mag_df[start_date:end_date]

    return mag_df

def run_test():
    pass

class filterbank:
    def __init__(self,):
        pass

    def build_fb_from_melbank(self,):
        pass

    def save_filterbank(self,):
        pass