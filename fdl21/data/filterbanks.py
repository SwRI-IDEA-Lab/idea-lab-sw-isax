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

import fdl21.data.prototyping_metrics as pm
import fdl21.utils.time_chunking as tc

omni_path = '/mnt/c/sw-data/nasaomnireader/'
year = '2019'
month = '05'


test_cdf_file_path =omni_path+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'


_MODEL_DIR = os.path.dirname( os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_SRC_DIR)

_PSP_MAG_DATA_DIR = '/sw-data/psp/mag_rtn/'
_WIND_MAG_DATA_DIR = '/sw-data/wind/mfi_h2/'
_OMNI_MAG_DATA_DIR = '/sw-data/nasaomnireader/'
_SRC_DATA_DIR = os.path.join(_SRC_DIR,'data',)

_EXPONENTS_LIST = [2.15, 1.05, 1.05]


def get_test_data(fname_full_path=None,
                  fname = None,
                  instrument = 'omni',
                  start_date = dt.datetime(year=2019,month=5,day=15,hour=0),
                  end_date = dt.datetime(year=2019,month=5,day=16,hour=0),
                  rads_norm=True,
                  orbit_fname = None,
                  return_sample_rate = False):
    """Retrieve a set of data to test and visualize filterbank application
    
    Parameters
    ----------
    fname_full_path : string
        complete file path to cdf file to extract data from
        A value for fname_full_path or fname (but not both) is required 
    fname : string
        part-way path to cdf file to extract data from, after 
        the selected "_DATA_DIR" that is selected by indicated instrument.
        A value for fname_full_path or fname (but not both) is required
    start_date: datetime, optional
        test data start time 
    end_date: datetime, optional
        test data end time
    rads_norm : bool, optional
        Boolean flag for controlling the normalization of the magnetic field 
        to account for the decay of the field strength with heliocentric distance
    orbit_fname : string, optional
        file path to psp orbit data
    """
    if fname_full_path is None:
        if instrument == 'psp':
            data_dir = _PSP_MAG_DATA_DIR
        elif instrument=='wind' :
            data_dir = _WIND_MAG_DATA_DIR
        elif instrument == 'omni':
            data_dir = _OMNI_MAG_DATA_DIR

        assert fname is not None, "Need to provide value for fname or fname_full_path"
        # Generate the full path to the file
        fname_full_path = os.path.join(
            _SRC_DIR + data_dir,
            *fname.split('/') # this is required do to behavior of os.join
        )
        
    if instrument == 'psp':
        if orbit_fname is not None:
            orbit_fname = os.path.join(
            _SRC_DATA_DIR,
            orbit_fname)
            # orbit dataframe
            orbit = pd.read_csv(
                orbit_fname,
                sep=",",
                comment ="#",
                index_col='EPOCH_yyyy-mm-ddThh:mm:ss.sssZ',
                parse_dates=['EPOCH_yyyy-mm-ddThh:mm:ss.sssZ'],
            )
        mag_df = pm.read_PSP_dataset(
            fname=fname_full_path,
            orbit=orbit,
            rads_norm=rads_norm,
            exponents_list=_EXPONENTS_LIST
        )
    elif instrument == 'wind':
        mag_df = pm.read_WIND_dataset(
            fname=fname_full_path
        )
    elif instrument == 'omni':
        mag_df = pm.read_OMNI_dataset(
            fname=fname_full_path
        )
    mag_df.interpolate(inplace=True)
    mag_df = mag_df[start_date:end_date]

    if return_sample_rate:
        avg_sampling_rate, _, _ = pm.check_sampling_freq(mag_df)
        return mag_df, avg_sampling_rate

    return mag_df

def add_DC_HF_filters(fb_matrix,
                      DC = True,
                      HF = True):
    # DC
    if DC:
        DC_filter = 1-fb_matrix[0,:]
        minin = (DC_filter == np.min(DC_filter)).nonzero()[0][0]
        DC_filter[minin:] = 0
        fb_matrix = np.append(DC_filter[None,:], fb_matrix, axis=0)

    # HF
    if HF:
        HF_filter = 1-fb_matrix[-1,:]
        minin = (HF_filter == np.min(HF_filter)).nonzero()[0][0]
        HF_filter[0:minin] = 0
        fb_matrix = np.append(fb_matrix, HF_filter[None,:], axis=0)

    return fb_matrix

def visualize_filterbank(fb_matrix,
                         fftfreq,
                         xlim:tuple = None,
                         melfreq = None):
    fig,ax = plt.subplots(figsize=(8,3))
    ax.plot(fftfreq,fb_matrix.T)
    ax.grid(True)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Frequency / Hz')
    if xlim is None:
        xlim = (np.min(fftfreq),np.max(fftfreq))
    ax.set_xlim(xlim)

    if melfreq is not None:
        ax2 = ax.twiny()
        ax2.xaxis.set_ticks_position('top')
        ax2.set_xlim(xlim)
        ax2.xaxis.set_ticks(melbank.mel_to_hertz(melfreq))
        ax2.xaxis.set_ticklabels(['{:.0f}'.format(mf) for mf in melfreq])
        ax2.set_xlabel('Frequency / mel')
    
    plt.tight_layout()
    plt.show()


def visualize_filterbank_application():
    pass

def run_test():
    pass

class filterbank:
    def __init__(self,):
        self.fb_matrix = None
        self.fftfreq = None
        self.melfreq = None
        self.n_bands = 0
        self.frequency_endpoints = None
        self.DC_HF = False

    def build_melbank_fb(self,
                         num_mel_bands = 2,
                         freq_min = 0,
                         freq_max = 5,
                         num_fft_bands = 100000,
                         sample_rate = 44100):
        """Build a filterbank, entirely using pyfilterbank's melbank 
        ([documentation](https://siggigue.github.io/pyfilterbank/melbank.html))
        
        **Note:** Traditional melbank filters are spread across the frequency spectrum  
        (on the *Mel* scale) in a way that is spaced lienarly at low frequencies 
        and logarithmically at higher frequencies. 
        """
        melmat, (melfreq,fftfreq) = melbank.compute_melmat(num_mel_bands=num_mel_bands,
                                                           freq_min=freq_min,
                                                           freq_max=freq_max,
                                                           num_fft_bands=num_fft_bands,
                                                           sample_rate=sample_rate)
        
        self.fb_matrix = melmat 
        self.fftfreq = fftfreq
        self.melfreq = melfreq
        self.n_bands = self.fb_matrix.shape[0]
    
    def build_manual_melbank(self,
                             frequency_endpoints:list = None,
                             fft_freq_range = (0,80000),
                             num_fft_bands = 100000):
        if frequency_endpoints is None:
            frequency_endpoints = [0.0,1.5,3.5,5.0]
        fftfreq = np.linspace(fft_freq_range[0],fft_freq_range[1],num_fft_bands)

        fb_matrix = []
        for i,endpt in enumerate(frequency_endpoints[:-1]):
            endpt_idx = (np.abs(fftfreq - endpt)).argmin()
            idx_p1 = (np.abs(fftfreq - frequency_endpoints[i+1])).argmin()

            pre_endpt = np.zeros(endpt_idx)

            n_pts = (idx_p1 - endpt_idx) + 1 # size of band "hill"
            up = np.linspace(0,1,n_pts//2)
            down = np.linspace(1,0,n_pts-len(up))
            
            filter = np.concatenate((pre_endpt,up,down),axis=0)

            post_endpt = np.zeros(len(fft_freq_range) - len(filter))

            filter = np.concatenate((filter,post_endpt),axis=0)

            fb_matrix.append(filter)
        
        self.fb_matrix = np.array(fb_matrix)
        self.fftfreq = fftfreq
        self.frequency_endpoints = frequency_endpoints
        self.melfreq = None

    def add_DC_HF_filters(self,
                          DC = True,
                          HF = True):
        self.fb_matrix = add_DC_HF_filters(fb_matrix=self.fb_matrix,
                                           DC=DC,
                                           HF=HF)
        self.n_bands = self.fb_matrix.shape[0]
        self.DC_HF = True
    
    def get_melbank_freq_endpoints(self):
        frequency_endpoints =[]
        # TODO (JK): Potentially need to come back and refine later
        for band in self.fb_matrix:
            idx = band.nonzero()[0][0]
            frequency_endpoints.append(self.fftfreq[idx])
        if not self.DC_HF:
            last_idx = self.fb_matrix[-1,:].nonzero()[0][-1]
            if last_idx+1 == len(self.fftfreq):
                frequency_endpoints.append(self.fftfreq[-1])
            else:
                frequency_endpoints.append(self.fftfreq[last_idx+1])
        else:
            last_idx = np.where(self.fb_matrix[-1,:]==1)[0][0]
            frequency_endpoints.append(self.fftfreq[last_idx])

        self.frequency_endpoints = np.array(frequency_endpoints)          
    
    def visualize_filterbank(self):
        self.get_melbank_freq_endpoints()
        visualize_filterbank(fb_matrix=self.fb_matrix,
                             fftfreq=self.fftfreq,
                             xlim=(self.frequency_endpoints[0],self.frequency_endpoints[-1]),
                             melfreq=self.melfreq)

    def save_filterbank(self,
                        save_path: str):

        self.get_melbank_freq_endpoints()
            
        filterbank_dictionary = {'fb_matrix': self.fb_matrix,
                                'fftfreq': self.fftfreq,
                                'frequencies': self.frequency_endpoints}
            
        with open(save_path + '_filterbank-dictionary.pkl', 'wb') as f:
            pickle.dump(filterbank_dictionary,f)

            