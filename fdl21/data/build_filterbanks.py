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

# %% Ch. 15 Formula
def moving_avg_freq_response(f,window=dt.timedelta(minutes=3000),cadence=dt.timedelta(minutes=1)):
    n = int(window.total_seconds()/cadence.total_seconds())
    numerator = np.sin(np.pi*f*n)
    denominator = n*np.sin(np.pi*f)
    return abs(numerator/denominator)

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
                         ylabel = 'Weight'):
    """Simple plot of filterbank"""
    fig,ax = plt.subplots(figsize=(8,3))
    ax.plot(fftfreq,fb_matrix.T)
    ax.grid(True)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_xlabel('Frequency  (Hz)')
    if xlim is None:
        xlim = (np.min(fftfreq),np.max(fftfreq))
    ax.set_xlim(xlim)
    
    plt.tight_layout()
    plt.show()

def visualize_filterbank_application(data_df,
                                     fb_matrix,
                                     fftfreq,
                                     data_col = None,
                                     cadence = dt.timedelta(seconds=300),
                                     figsize=(24,8),
                                     wordsize_factor=2,
                                     gs_wspace = 0.2,
                                     gs_hspace = 0,
                                     xlim = None,
                                     center_freq = None,
                                     DC = False,
                                     HF = False,
                                     show_plot=True,
                                     save_results=False):
    """Plot comprehensive visualization of filterbank and its application to a set of test data.
    Plot includes the filterbank, raw test data, decomposition of filterbank preprocessed data and PAA, 
    and series recovered from summing up each filterbank PAA application.
    
    Parameters
    ----------
    
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(ncols = 3, nrows = fb_matrix.shape[0]*2,
                          figure = fig,
                          wspace=gs_wspace, hspace=gs_hspace)
    
    if data_col is None:
        data_col = data_df.columns[-1]
    x = data_df.index
    y = data_df[data_col]
    all_filtered = np.zeros((fb_matrix.shape[0],data_df.shape[0]))
    all_paa = np.zeros((fb_matrix.shape[0],data_df.shape[0]))
    total_paa = np.zeros(data_df[data_col].shape)

    data_span = x[-1]-x[0]

    
    for i in range(fb_matrix.shape[0]):
        # get filtered signal
        filtered_sig = tc.preprocess_fft_filter(mag_df=data_df,
                                                cols=data_df.columns,
                                                cadence=cadence,
                                                frequency_weights=fb_matrix[i,:],
                                                frequency_spectrum=fftfreq)
        
        filtered_sig = np.array(filtered_sig[data_col])
        
        all_filtered[i] = filtered_sig

        # wordsize calculation
        if HF and i == fb_matrix.shape[0]-1:
            word_size = int(0.9*len(x))
        else:
            if DC and i == 0:
                freq = (center_freq[1]-center_freq[0])/2
            else:
                freq = center_freq[i]
            word_size = int(wordsize_factor*data_span.total_seconds()*freq)

            if word_size > len(x):
                word_size = len(x)
        
        # PAA application
        paa = PiecewiseAggregateApproximation(word_size)
        paa_sequence = paa.fit_transform(filtered_sig.reshape(1,-1))

        paa_sfull = paa.inverse_transform(paa_sequence)
        paa_sfull = paa_sfull.reshape(-1)

        all_paa[i] = paa_sfull
        total_paa = total_paa + paa_sfull 

        ax0 = fig.add_subplot(gs[2*i:2*i+2,1])    
        ax0.plot(x, filtered_sig,label=f'center_freq = {center_freq[i]:.2e}')
        # ax0.plot(x, paa_sfull, c='r',label=f'word_size = {word_size}')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.legend(loc='upper right',bbox_to_anchor=(1.4, 1))

        if i==0:
            ax0.set_title('Filter bank decomposition')
        
    if fb_matrix.shape[0]<5:
        os_gs = gs[3:4,0]
        pr_gs = gs[5:6,0]
    else:
        os_gs = gs[4:6,0]
        pr_gs = gs[7:9,0]
    # ax0 = fig.add_subplot(pr_gs)   
    # ax0.plot(x, total_paa, c='r')
    # ax0.set_title('Series recovered from filter bank PAA')
    # ax0.set_xticks([])
    # ax0.set_yticks([])


    ax0 = fig.add_subplot(os_gs)   
    ax0.plot(x, y)
    ax0.set_title('Original series')
    # ax0.plot(x[0:-1:20], total_paa[0:-1:20], c='r')
    ax0.set_xticks([])
    ax0.set_yticks([])

    if xlim is None:
        xlim = (fftfreq[0],fftfreq[-1])
    ax = fig.add_subplot(gs[0:2,0])  
    ax.plot(fftfreq, fb_matrix.T)
    ax.grid(True)
    # ax.set_ylabel('Weight')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim(xlim)
    ax.set_title('Mel filter bank')
    ax.set_xticks(center_freq)
    ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    if show_plot:
        plt.show()
    else:
        plt.close()

    if save_results:
        return all_filtered,all_paa

class filterbank:
    def __init__(self,
                 data_len:int,
                 cadence = dt.timedelta(seconds=60),
                 restore_from_file:str = None):
        self.data_len = data_len
        self.cadence = cadence
        self.freq_spectrum = np.linspace(0.0001,data_len/2,(data_len//2)+1)
        self.freq_smprt = np.linspace(0.0001,0.5,(data_len//2)+1)
        self.freq_hz_spec = self.freq_spectrum/(data_len*cadence.total_seconds())
        
        # placeholders
        self.fb_matrix = None
        self.edge_freq = None
        self.DC = False
        self.HF = False

        # if restore_from_file is not None:
        #     pkl = open(restore_from_file,'rb')
        #     fb_dict = pickle.load(pkl)
        #     self.fb_matrix = fb_dict['fb_matrix']
        #     self.fftfreq = fb_dict['fftfreq']
        #     self.edge_freq = fb_dict['edge_freq']
        #     self.DC = fb_dict['DC']
        #     self.HF = fb_dict['HF']

    def build_triangle_fb(self, 
                          filter_freq_range = (0,5),
                          num_bands = 2,
                          center_freq = None):
        """Creates filterbank matrix of triangle filters.

        Parameters
        ----------
        filter_freq_range : tuple
            (min_freq,max_freq)
            Minimum and maximum frequencies (in hz) that define the range in which the filters occupy.
            min_freq will be the first edge, and max_freq will be the last edge
            
        num_bands : int
            Number of filters in filter bank. 
            Used to evenly space out center frequencies across filter_freq_range.
            Not used if center_freq is not None
        center_freq : array
            Specified center frequencies of triangle filterbanks.
            If none or empty array, center_freq of the filterbank will be evenly spaced out using num_bands. 
        """
        freq_min, freq_max = filter_freq_range

        # if center frequencies not specified, centers are evenly spaced out given the 
        if center_freq is None or len(center_freq) == 0:
            delta_freq = abs(freq_max - freq_min) / (num_bands + 1.0)
            edge_freq = freq_min + delta_freq*arange(0, num_bands+2)
            center_freq = edge_freq[1:-1]
        else:
            if type(center_freq) == np.ndarray:
                center_freq = center_freq.tolist()
            edge_freq = [freq_min] + center_freq + [freq_max]
            num_bands = len(center_freq)
        
        lower_edges = edge_freq[:-2]
        upper_edges = edge_freq[2:]
        

        freqs = self.freq_hz_spec
        melmat = zeros((num_bands, len(freqs)))

        for iband, (center, lower, upper) in enumerate(zip(
                center_freq, lower_edges, upper_edges)):

            left_slope = (freqs >= lower)  == (freqs <= center)
            melmat[iband, left_slope] = (
                (freqs[left_slope] - lower) / (center - lower)
            )

            right_slope = (freqs >= center) == (freqs <= upper)
            melmat[iband, right_slope] = (
                (upper - freqs[right_slope]) / (upper - center)
            )
        self.fb_matrix = melmat 
        self.edge_freq = np.array(edge_freq)
        self.upper_edges = upper_edges
        self.center_freq = center_freq
        self.lower_edges = lower_edges

    def build_DTSM_fb(self,
                      windows = []):
        fb_matrix = zeros((len(windows)-1,len(self.freq_spectrum)))
        center_freq = []
        windows.sort(reverse=True)
        for i,w in enumerate(windows[:-1]):
            DT = 1 - moving_avg_freq_response(f=self.freq_smprt,
                                              window=dt.timedelta(seconds=w),
                                              cadence=self.cadence)
            SM = moving_avg_freq_response(f=self.freq_smprt,
                                          window=dt.timedelta(seconds=windows[i+1]),
                                          cadence=self.cadence)
            FR = SM*DT
            fb_matrix[i] = FR
            center_freq.append(self.freq_hz_spec[np.argmax(FR)])
        self.fb_matrix = fb_matrix
        self.center_freq = center_freq
        self.windows = windows
        

    def add_DC_HF_filters(self,
                          DC = True,
                          HF = True):
        self.fb_matrix = add_DC_HF_filters(fb_matrix=self.fb_matrix,
                                           DC=DC,
                                           HF=HF)
        if DC:
            if self.center_freq[0] != self.edge_freq[0]:
                self.center_freq = np.insert(self.center_freq,0,self.edge_freq[0])
            if self.upper_edges[0] != self.edge_freq[1]:
                self.upper_edges = np.insert(self.upper_edges,0,self.edge_freq[1])
        if HF:
            if self.center_freq[-1] != self.edge_freq[-1]:
                self.center_freq = np.append(self.center_freq,self.edge_freq[-1])
            if self.lower_edges[-1] != self.edge_freq[-2]:
                self.lower_edges = np.append(self.lower_edges,self.edge_freq[-2])
        self.DC = DC
        self.HF = HF

    def add_mvgavg_DC_HF(self,
                         DC = True,
                         HF = True):
        if DC:
            SM = moving_avg_freq_response(f=self.freq_spectrum,
                                            window=dt.timedelta(seconds=max(self.windows)),
                                            cadence=self.cadence)
            self.fb_matrix = np.append(SM[None,:],self.fb_matrix,axis=0)
            self.DC = True
            cnt_fq = self.freq_hz_spec[np.argmax(SM)]
            if self.center_freq[0] != cnt_fq:
                self.center_freq = np.insert(self.center_freq,0,cnt_fq)
            # if self.center_freq[-1] != cnt_fq:
            #     self.center_freq = np.append(self.center_freq,cnt_fq)

        if HF:
            FR = moving_avg_freq_response(f=self.freq_spectrum,
                                            window=dt.timedelta(seconds=min(self.windows)),
                                            cadence=self.cadence)
            DT = 1 - FR
            self.fb_matrix = np.append(self.fb_matrix,DT[None,:],axis=0)
            self.HF = True
            for i,f in enumerate(FR[:-1]):
                if f - FR[i+1] <0:
                    cnt_fr_idx = i
                    break
            cnt_fq = self.freq_hz_spec[cnt_fr_idx]
            if self.center_freq[-1] != cnt_fq:
                self.center_freq = np.append(self.center_freq,cnt_fq)
            # if self.center_freq[0] != cnt_fq:
            #     self.center_freq = np.insert(self.center_freq,0,cnt_fq)


    def visualize_filterbank(self):
        """Show a plot of the built filterbank."""
        visualize_filterbank(fb_matrix=self.fb_matrix,
                             fftfreq=self.freq_hz_spec,)
                            #  xlim=(self.edge_freq[0],self.edge_freq[-1]))

    # TODO: Update and fix filterbank saving with new updates (changed attributes, moving average FB, etc.)
    def save_filterbank(self):
        """Save the filterbank transformation matrix, fftfrequencies, and frequency endpoints 
        as a dictionary to a local pickle file"""
            
        filterbank_dictionary = {'fb_matrix': self.fb_matrix,
                                'fftfreq': self.fftfreq,
                                'edge_freq': self.edge_freq,
                                'center_freq': self.center_freq,
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
    mag_df = mag_df-mag_df.mean()
    # TODO: Set up json excutable way to test (using args, etc.)
    #=====================================
    # fb = filterbank()
    # fb.build_triangle_fb()
    # fb.add_DC_HF_filters()
    # fb.visualize_filterbank()
    #=====================================

    #=====================================
    fb = filterbank(data_len=len(mag_df),
                    cadence=dt.timedelta(seconds=60))
    fb.build_triangle_fb(num_bands=4,
                        filter_freq_range=(0.0,0.001),
                        )
    visualize_filterbank(fb_matrix=fb.fb_matrix,
                         fftfreq=fb.freq_hz_spec,
                         xlim=(fb.edge_freq[0],fb.edge_freq[-1]),
                         ylabel='Amplitude')
    fb.add_DC_HF_filters()
    # fb.visualize_filterbank()
    visualize_filterbank(fb_matrix=fb.fb_matrix,
                         fftfreq=fb.freq_hz_spec,
                         xlim=(fb.edge_freq[0],fb.edge_freq[-1]),
                         ylabel='Amplitude')
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

    #=====================================
    # fb = filterbank(data_len=len(mag_df),
    #                 cadence=dt.timedelta(seconds=60))
    # fb.build_DTSM_fb(windows=[1000,3000,18000,108000])
    # fb.visualize_filterbank()
    # fb.add_mvgavg_DC_HF()
    # fb.visualize_filterbank()
    # visualize_filterbank_application(data_df=mag_df,
    #                                  fb_matrix=fb.fb_matrix,
    #                                  fftfreq=fb.freq_hz_spec,
    #                                  data_col='BY_GSE',
    #                                  cadence=dt.timedelta(minutes=1),
    #                                  wordsize_factor = 3,
    #                                  xlim = (0,0.001),
    #                                  center_freq = fb.center_freq,
    #                                  DC=fb.DC,
    #                                  HF=fb.HF)
    #=====================================

    visualize_filterbank_application(data_df=mag_df,
                                     fb_matrix=fb.fb_matrix,
                                     fftfreq=fb.freq_hz_spec,
                                     data_col='BY_GSE',
                                     cadence=dt.timedelta(minutes=1),
                                     wordsize_factor = 3,
                                     xlim = (fb.edge_freq[0],fb.edge_freq[-1]),
                                     center_freq = fb.center_freq,
                                     DC=fb.DC,
                                     HF=fb.HF)