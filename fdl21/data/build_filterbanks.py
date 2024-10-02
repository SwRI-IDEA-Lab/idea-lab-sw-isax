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

_FILE_DIR = os.path.dirname( os.path.abspath(__file__))
_MODEL_DIR = os.path.dirname(_FILE_DIR)
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_SRC_DIR)

# local imports
import fdl21.data.prototyping_metrics as pm
import fdl21.utils.time_chunking as tc

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
                  orbit_fname = None):
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


    return mag_df

def build_triangle_filterbank(num_bands=12,
                            frequencies = [],
                            freq_range = (0,5),
                            num_fft_bands=513, 
                            sample_rate=16000):
    """Returns tranformation matrix for mel spectrum.

    Parameters
    ----------
    num_mel_bands : int
        Number of mel bands. Number of rows in melmat.
        Default: 24
    freq_min : scalar
        Minimum frequency for the first band.
        Default: 64
    freq_max : scalar
        Maximum frequency for the last band.
        Default: 8000
    num_fft_bands : int
        Number of fft-frequency bands. This ist NFFT/2+1 !
        number of columns in melmat.
        Default: 513   (this means NFFT=1024)
    sample_rate : scalar
        Sample rate for the signals that will be used.
        Default: 44100

    Returns
    -------
    melmat : ndarray
        Transformation matrix for the mel spectrum.
        Use this with fft spectra of num_fft_bands_bands length
        and multiply the spectrum with the melmat
        this will tranform your fft-spectrum
        to a mel-spectrum.

    frequencies : tuple (ndarray <num_mel_bands>, ndarray <num_fft_bands>)
        Center frequencies of the mel bands, center frequencies of fft spectrum.

    """
    if len(frequencies) == 0:
        freq_min, freq_max = freq_range
        delta_freq = abs(freq_max - freq_min) / (num_bands + 1.0)
        frequencies = freq_min + delta_freq*arange(0, num_bands+2)
    assert len(frequencies) == num_bands + 2, "frequencies must have length num_bands + 2"
    lower_edges = frequencies[:-2]
    upper_edges = frequencies[2:]
    center_frequencies = frequencies[1:-1]

    freqs = linspace(0.0, sample_rate/2.0, num_fft_bands)
    melmat = zeros((num_bands, num_fft_bands))

    for iband, (center, lower, upper) in enumerate(zip(
            center_frequencies, lower_edges, upper_edges)):

        left_slope = (freqs >= lower)  == (freqs <= center)
        melmat[iband, left_slope] = (
            (freqs[left_slope] - lower) / (center - lower)
        )

        right_slope = (freqs >= center) == (freqs <= upper)
        melmat[iband, right_slope] = (
            (upper - freqs[right_slope]) / (upper - center)
        )

    return melmat, freqs, (frequencies,lower_edges,center_frequencies, upper_edges)


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
                         xlim:tuple = None):
    """Simple plot of filterbank"""
    fig,ax = plt.subplots(figsize=(8,3))
    ax.plot(fftfreq,fb_matrix.T)
    ax.grid(True)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Frequency  (Hz)')
    if xlim is None:
        xlim = (np.min(fftfreq),np.max(fftfreq))
    ax.set_xlim(xlim)
    
    plt.tight_layout()
    plt.show()

def visualize_filterbank_application(data_df,
                                     melmat,
                                     fftfreq,
                                     data_col = None,
                                     cadence = dt.timedelta(seconds=300),
                                     figsize=(24,8),
                                     wordsize_factor=2,
                                     gs_wspace = 0.2,
                                     gs_hspace = 0,
                                     xlim = None,
                                     edge_freq = None,
                                     DC = False,
                                     HF = False
                                     ):
    """Plot comprehensive visualization of filterbank and its application to a set of test data.
    Plot includes the filterbank, raw test data, decomposition of filterbank preprocessed data and PAA, 
    and series recovered from summing up each filterbank PAA application.
    
    Parameters
    ----------
    
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(ncols = 3, nrows = melmat.shape[0]*2,
                          figure = fig,
                          wspace=gs_wspace, hspace=gs_hspace)
    
    if data_col is None:
        data_col = data_df.columns[-1]
    x = data_df.index
    y = data_df[data_col]
    total = np.zeros(data_df[data_col].shape)
    total_paa = np.zeros(data_df[data_col].shape)

    data_span = x[-1]-x[0]

    
    for i in range(melmat.shape[0]):
        filtered_sig = tc.preprocess_fft_filter(mag_df=data_df,
                                                cols=data_df.columns,
                                                cadence=cadence,
                                                frequency_weights=melmat[i,:],
                                                frequency_spectrum=fftfreq)
        
        filtered_sig = np.array(filtered_sig[data_col])
        
        total = total + filtered_sig

        # TODO: Utilize center_frequencies, instead of edge_freq and if/else DC statement?
        freq_idx = i if DC else i+1
        word_size = int(wordsize_factor*data_span.total_seconds()*edge_freq[freq_idx])
        if word_size > len(x):
            word_size = len(x)
        paa = PiecewiseAggregateApproximation(word_size)
        paa_sequence = paa.fit_transform(filtered_sig[None,:])

        paa_sfull = paa.inverse_transform(paa_sequence)[0].ravel()

        total_paa = total_paa + paa_sfull 

        ax0 = fig.add_subplot(gs[2*i:2*i+2,1])    
        ax0.plot(x, filtered_sig)
        ax0.plot(x, paa_sfull, c='r',label=f'word_size = {word_size}')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.legend()

        if i==0:
            ax0.set_title('Filter bank decomposition')
        

    ax0 = fig.add_subplot(gs[3:5,2])   
    ax0.plot(x, total_paa, c='r')
    ax0.set_title('Series recovered from filter bank PAA')
    ax0.set_xticks([])
    ax0.set_yticks([])


    ax0 = fig.add_subplot(gs[4:6,0])   
    ax0.plot(x, y-np.mean(y))
    ax0.set_title('Original series')
    # ax0.plot(x[0:-1:20], total_paa[0:-1:20], c='r')
    ax0.set_xticks([])
    ax0.set_yticks([])

    if xlim is None:
        xlim = (fftfreq[0],fftfreq[-1])
    ax = fig.add_subplot(gs[0:2,0])  
    ax.plot(fftfreq, melmat.T)
    ax.grid(True)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim(xlim)
    ax.set_title('Mel filter bank')
    ax.set_xticks(edge_freq)
    ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    plt.show()

class filterbank:
    def __init__(self,
                 restore_from_file:str = None):
        self.fb_matrix = None
        self.fftfreq = None
        self.edge_freq = None
        self.DC = False
        self.HF = False

        if restore_from_file is not None:
            pkl = open(restore_from_file,'rb')
            fb_dict = pickle.load(pkl)
            self.fb_matrix = fb_dict['fb_matrix']
            self.fftfreq = fb_dict['fftfreq']
            self.edge_freq = fb_dict['edge_freq']
            self.DC = fb_dict['DC']
            self.HF = fb_dict['HF']

    def build_triangle_fb(self,
                         num_bands = 2,
                         frequencies = [],
                         freq_range = (0,5),
                         num_fft_bands=513, 
                         sample_rate=16000):
        """Build a filterbank, entirely using pyfilterbank's melbank 
        ([documentation](https://siggigue.github.io/pyfilterbank/melbank.html))
        
        **Note:** Traditional melbank filters are spread across the frequency spectrum  
        (on the *Mel* scale) in a way that is spaced lienarly at low frequencies 
        and logarithmically at higher frequencies. 
        """
        melmat, fftfreq, (frequencies,lower_edges,center_frequencies,upper_edges) = build_triangle_filterbank(num_bands=num_bands,
                                                                                                    frequencies=frequencies,
                                                                                                    freq_range=freq_range,
                                                                                                    num_fft_bands=num_fft_bands, 
                                                                                                    sample_rate=sample_rate)
        
        self.fb_matrix = melmat 
        self.fftfreq = fftfreq
        self.edge_freq = np.array(frequencies)
        self.upper_edges = upper_edges
        self.center_frequencies = center_frequencies
        self.lower_edges = lower_edges

    def add_DC_HF_filters(self,
                          DC = True,
                          HF = True):
        self.fb_matrix = add_DC_HF_filters(fb_matrix=self.fb_matrix,
                                           DC=DC,
                                           HF=HF)
        if DC:
            if self.center_frequencies[0] != self.edge_freq[0]:
                self.center_frequencies = np.insert(self.center_frequencies,0,self.edge_freq[0])
            if self.upper_edges[0] != self.edge_freq[1]:
                self.upper_edges = np.insert(self.upper_edges,0,self.edge_freq[1])
        if HF:
            if self.center_frequencies[-1] != self.edge_freq[-1]:
                self.center_frequencies = np.append(self.center_frequencies,self.edge_freq[-1])
            if self.lower_edges[-1] != self.edge_freq[-2]:
                self.lower_edges = np.append(self.lower_edges,self.edge_freq[-2])
        self.DC = DC
        self.HF = HF
    
    def visualize_filterbank(self):
        """Show a plot of the built filterbank."""
        visualize_filterbank(fb_matrix=self.fb_matrix,
                             fftfreq=self.fftfreq,
                             xlim=(self.edge_freq[0],self.edge_freq[-1]))

    def save_filterbank(self):
        """Save the filterbank transformation matrix, fftfrequencies, and frequency endpoints 
        as a dictionary to a local pickle file"""
            
        filterbank_dictionary = {'fb_matrix': self.fb_matrix,
                                'fftfreq': self.fftfreq,
                                'edge_freq': self.edge_freq,
                                'center_frequencies': self.center_frequencies,
                                'lower_edges': self.lower_edges,
                                'upper_edges': self.upper_edges,
                                'DC': self.DC,
                                'HF': self.HF
                                }
            
       
        fb_prefix = f'fb'
        for edge in self.edge_freq:
            fb_prefix +=f'_{edge:.3e}'
        if self.DC:
            fb_prefix += '_DC'
        if self.HF:
            fb_prefix += '_HF'

        with open(_SRC_DATA_DIR + '/filterbanks/' + fb_prefix +'.pkl', 'wb') as f:
            pickle.dump(filterbank_dictionary,f)

if __name__ == '__main__':
    year = '2019'
    month = '05'
    test_cdf_file_path =_SRC_DIR+_OMNI_MAG_DATA_DIR+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

    mag_df = get_test_data(fname_full_path=test_cdf_file_path)

    #=====================================
    # fb = filterbank()
    # fb.build_triangle_fb()
    # fb.add_DC_HF_filters()
    # fb.visualize_filterbank()
    #=====================================

    #=====================================
    fb = filterbank()
    fb.build_triangle_fb(num_bands=7,
                        sample_rate=1/60,
                        freq_range=(0.0,0.001),
                        num_fft_bands=int(1E6))
    fb.add_DC_HF_filters()
    fb.visualize_filterbank()
    #=====================================

    #=====================================
    # fb = filterbank()
    # fb.build_triangle_fb(num_bands=4,
    #                     sample_rate=1/60,
    #                     frequencies=[0.0,0.00025,0.00037,0.00065,0.000828,0.001],
    #                     num_fft_bands=int(1E6))
    # # fb.add_DC_HF_filters()
    # fb.visualize_filterbank()
    #=====================================

    #=====================================
    # fb = filterbank(restore_from_file='/home/jkobayashi/gh_repos/idea-lab-sw-isax/data/filterbanks/fb_0.000e+00_1.250e-04_2.500e-04_3.750e-04_5.000e-04_6.250e-04_7.500e-04_8.750e-04_1.000e-03.pkl')
    # fb.visualize_filterbank()
    #=====================================

    visualize_filterbank_application(data_df=mag_df,
                                     melmat=fb.fb_matrix,
                                     fftfreq=fb.fftfreq,
                                     data_col='BY_GSE',
                                     cadence=dt.timedelta(minutes=1),
                                     wordsize_factor = 3,
                                     xlim = (fb.edge_freq[0],fb.edge_freq[-1]),
                                     edge_freq = fb.edge_freq,
                                     DC=fb.DC,
                                     HF=fb.HF)