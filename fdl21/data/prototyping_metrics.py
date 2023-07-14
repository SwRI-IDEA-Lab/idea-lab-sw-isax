"""
This module contains a list of functions that were prototyped 
in the wind_visualization.ipynb. The purpose of this module is 
to encapsulate the key functions developed in an interactive manner
in the jupyter notebook, as well as added the necessary docstrings 
and desired formatting. 
"""
import datetime as dt
import logging
import os
import sys
import glob

mod_dir = os.path.abspath(__file__).split('/')[:-2]
src_directory = '/'.join(mod_dir)
sys.path.append(src_directory)

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

import cdflib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import fdl21.utils.time_chunking as time_chunking


def convert_WIND_EPOCH(cdf):
    """ Convenience function for generating datetimes from WIND data

    The EPOCH used here is the time in miliseconds that
    has elapsed since 0000-01-01. Compute this by converting
    the units of time from miliseconds to days and 
    then adding that to the first date (0000-01-01).

    Parameters
    ----------
    
    cdf : cdflib.CDF
        CDF object that encapsulates the dataset we are analyzing 
    
    Returns
    -------
    dates : np.array
        A numpy array of datetime objects
    
    """
    # Read in time data
    time_ms = cdf['EPOCH']
   
    # Convert to time in seconds
    time_s = time_ms * 1e-3
    
    # Compute the number of days and store it as dt.timedelta object
    # Note that this only works because the first argument to the 
    # dt.timedelta object is a numerical value that represents the
    # number of days.
    time_days = np.array(
        list(map(dt.timedelta, time_s * (1 / (24 * 3600))))
    )
    
    # Python does not handle year 0, so we have to start at 1 AD
    year0 = dt.datetime(year=1, month=1, day=1)
    
    # Compute the date by adding the total time in days to the first year
    # For some reason, we need to add one extra day or else we're off by 1
    dates = year0 + time_days - dt.timedelta(days=366)

    return dates


def convert_PSP_EPOCH(cdf):
    """ Convenience function for generating datetimes from PSP data

    The EPOCH used here is the time in miliseconds that
    has elapsed since 0000-01-01. Compute this by converting
    the units of time from miliseconds to days and 
    then adding that to the first date (0000-01-01).

    Parameters
    ----------
    
    cdf : cdflib.CDF
        CDF object that encapsulates the dataset we are analyzing 
    
    Returns
    -------
    dates : np.array
        A numpy array of datetime objects
    
    """
    # Read in epoch data and return datetimes
    return cdflib.epochs.CDFepoch.to_datetime(cdf['epoch_mag_RTN'])

    
def check_sampling_freq(mag_df, min_sep=None, verbose=False):
    """Determine the sampling frequency from the data

    Compute a weighted-average of the sampling frequency
    present in the time-series data. This is done by taking
    the rolling difference between consecutive datetime indices
    and then binning them up using a method of pd.Series objects.
    Also computes some statistics describing the distribution of
    sampling frequencies.

    Parameters
    ----------
    mag_df : pd.DataFrame
        Pandas dataframe containing the magnetometer data

    min_sep : float
        Minimum separation between two consecutive observations 
        to be consider usable for discontinuity identification

    verbose : boolean
        Specifies information on diverging sampling frequencies
    
    Returns
    -------
    avg_sampling_freq : float
        Weighted average of the sampling frequencies in the dataset

    stats : dict
        Some descriptive statistics for the interval
    
    """
    # Boolean flag for quality of data in interval
    # Assume its not bad and set to True if it is
    bad = False

    # Compute the time difference between consecutive measurements
    # a_i - a_{i-1} and save the data as dt.timedelta objects
    # rounded to the nearest milisecond
    diff_dt = mag_df.index.to_series().diff(1).round('ms')
    sampling_freqs = diff_dt.value_counts()
    sampling_freqs /= sampling_freqs.sum()

    avg_sampling_freq = 0
    for t, percentage in sampling_freqs.items():
        avg_sampling_freq += t.total_seconds() * percentage

    # Compute the difference in units of seconds so we can compute the RMS
    diff_s = np.array(
                list(
                    map(lambda val: val.total_seconds(), diff_dt)
                )
            )

    # Compute the RMS of the observation times to look for gaps in 
    # in the observation period
    t_rms = np.sqrt(
                np.nanmean(
                    np.square(diff_s)
                )
            )
    # flag that the gaps larger the min_sep. 
    if min_sep is None:
        min_sep = 5 * t_rms

    gap_indices = np.where(diff_s > min_sep)[0]
    n_gaps = len(gap_indices)
    
    try:
        previous_indices = gap_indices - 1
    except TypeError as e:
        LOG.warning(e)
        total_missing = 0
    else:
        interval_durations = mag_df.index[gap_indices] \
                                - mag_df.index[previous_indices]
        total_missing = sum(interval_durations.total_seconds())

    # Compute the duration of the entire interval and determine the coverage
    total_duration = (mag_df.index[-1] - mag_df.index[0]).total_seconds()
    coverage = 1 - total_missing / total_duration

    if verbose and coverage < 0.5:
        msg = (
            f"\n Observational coverage: {coverage:0.2%}\n"
            f"Number of data gaps: {n_gaps:0.0f}\n"
            f"Average sampling rate: {avg_sampling_freq:0.5f}"
            )
        LOG.warning(msg)
        bad = True

    stats_data = {}
    stats_data['average_freq'] = avg_sampling_freq
    stats_data['max_freq'] = sampling_freqs.index.max().total_seconds()
    stats_data['min_freq'] = sampling_freqs.index.min().total_seconds()
    stats_data['n_gaps'] = len(gap_indices)
    stats_data['starttime_gaps'] = [mag_df.index[previous_indices]]
    stats_data['total_missing'] = total_missing
    stats_data['coverage'] = coverage

    return avg_sampling_freq, stats_data, bad


def compute_B_field_stats(mag_df, cols=None):
    """Compute statistics on the magnetic field

    Parameters
    ----------
    mag_df : pd.DataFrame
        Dataframe containing magnetometer data

    cols : list
        List of column names to process

    Returns
    -------
    stats_data : dict
        `dict` containing statistics on the magnetic field
    """
    if cols is None:
        cols = mag_df.columns

    stats_data = {}
    for col in cols:
        mean = np.nanmean(mag_df[col])
        min_val = np.nanmin(mag_df[col])
        max_val = np.nanmax(mag_df[col])
        rms = np.sqrt(
                np.nanmean(
                    np.square(mag_df[col] - mean)
                )
            )
        stats_data[f'{col}_mean'] = mean
        stats_data[f'{col}_rms'] = rms
        stats_data[f'{col}_min'] = min_val
        stats_data[f'{col}_max'] = max_val
    
    return stats_data

def compute_B_diff_stats(mag_df, cols=None):
    """Compute statistics on the difference between datapoints of the magne  field

    Parameters
    ----------
    mag_df : pd.DataFrame
        Dataframe containing magnetometer data

    cols : list
        List of column names to process
    
    Returns
    -------
    stats_data : dict
        `dict` containing statistics on the magnetic field
    """
    if cols is None:
        cols = mag_df.columns

    diff_BF = mag_df.diff(axis=0)
    stats_data = {}
    for col in cols:
        mean = np.nanmean(diff_BF[col])
        min_val = np.nanmin(diff_BF[col])
        max_val = np.nanmax(diff_BF[col])
        rms = np.sqrt(
                np.nanmean(
                    np.square(diff_BF[col] - mean)
                )
            )
        stats_data[f'{col}_diff_mean'] = mean
        stats_data[f'{col}_diff_rms'] = rms
        stats_data[f'{col}_diff_min'] = min_val
        stats_data[f'{col}_diff_max'] = max_val
    
    return stats_data

def psp_rad_norm(psp_tmp_mag, psp_time, orbit, exponents_list):
    """
    Normalizes the PSP magnetic field data with respect to radial distance from Sun
    Parameters
    ----------
    psp_tmp_mag : pandas.DataFrame
        pandas.DataFrame that contains the magnetic field data
    psp_time : datetime
        Time of measurements
    orbit : pd.DataFrame
        Dataframe with instrument's orbit
    exponents_list : np.array
        Numpy array containing exponentials from curve fits to magnetic field vs heliocentric distance
    Returns
    -------
    stats_data : dict
        `dict` containing statistics on the magnetic field
    """
    # Compute radius of orbit
    radius = interpolate_orbit(psp_time, orbit)['RAD_AU_AU']
    # Rescale the magnetic field based on radial decay (taken from exponents_list)
    for i in range(0,3):
        psp_tmp_mag[:,i] = psp_tmp_mag[:,i]*np.power(radius, exponents_list[i])
    return psp_tmp_mag

def read_PSP_dataset(fname, orbit=None, rads_norm=False, exponents_list=None):
    """Function for reading in the PSP dataset
    
    Parameters
    ----------
    fname : str
        Filename of dataset to read in
    
    orbit : pd.DataFrame
        Dataframe with instrument's orbit
    
    rads_norm : bool
        Flag that triggers the normalization of the magnetic field according to the 
        distance to the Sun

    exponents_list : np.array
        Provides list of exponents for each mgn component to normalize the mgn field with respect 
        to the radial distance from the Sun
    
    Returns
    -------
    mag_df : pd.DataFrame
        Pandas dataframe containing the magnetometer data
    """
    cdf_file = cdflib.CDF(fname)
    psp_tmp_mag = cdf_file['psp_fld_l2_mag_RTN']
    data_quality = cdf_file['psp_fld_l2_quality_flags']

    if rads_norm:
        psp_time = pd.DataFrame(
            index=cdflib.epochs.CDFepoch.to_datetime(cdf_file['epoch_mag_RTN'])
        )
        psp_tmp_mag = psp_rad_norm(
            psp_tmp_mag=psp_tmp_mag,
            psp_time=psp_time.index,
            orbit=orbit,
            exponents_list=exponents_list
        )
            
    #LOG.info(data_quality.shape)
    #LOG.info(data_quality)

    # Imprint bad quality time-stamps as NaNs on Magnetic field components
    # psp_tmp_mag[data_quality!=0,:] = np.nan

    mag_data = {}

    # Calculating the magnetic field magnitude from PSP mag_RTN
    mag_data['B_mag'] = np.sqrt(
        (psp_tmp_mag[:,0])**2 + (psp_tmp_mag[:,1])**2, (psp_tmp_mag[:,2])**2
    )
    for i in range(3):
        mag_data[f'BRTN_{i}'] = psp_tmp_mag[:,i]
     

    dates = convert_PSP_EPOCH(cdf_file)
    mag_df = pd.DataFrame(mag_data, index=pd.DatetimeIndex(dates)).sort_index()
    return mag_df 


def read_WIND_dataset(fname):
    """Function for reading the WIND the dataset
    
    Parameters
    ----------
    fname : str
        Filename of dataset to read in
    
    Returns
    -------
    mag_df : pd.DataFrame
        Pandas dataframe containing the magnetometer data
    """
    cdf_file = cdflib.CDF(fname)
    mag_data = {}
    mag_data['B_mag'] = cdf_file['BF1']
    for i in range(3):
        mag_data[f'BGSE_{i}'] = cdf_file['BGSE'][:,i]
    
    dates = convert_WIND_EPOCH(cdf_file)
    mag_df = pd.DataFrame(mag_data, index=pd.DatetimeIndex(dates)).sort_index()
    return mag_df 

def read_OMNI_dataset(dir_name):
    """Function for reading the OMNI the dataset
    
    Parameters
    ----------
    dir_name : str
        Directory name that contains OMNI lst and fmt files
    
    Returns
    -------
    mag_df : pd.DataFrame
        Pandas dataframe containing the magnetometer data
    """
    # Open files
    lst_fname = glob.glob(f'{dir_name}/*lst')[0]
    fmt_fname = glob.glob(f'{dir_name}/*fmt')[0]

    omni_file = open(lst_fname)
    omni_label = open(fmt_fname)
    omni_label = omni_label.readlines()[4:] # skips first four lines which don't have data

    # Extracting the column names from fmt file
    columns = []                                           #list of column names to create
    for row in omni_label:
            columns.append(row.split()[1].replace(',',''))         #column names = [1] element of each line in fmt file
                                                                            # and remove commas

    #Create dictionary of OMNI data, then to dataframe
    omni_dict={key:[] for key in columns}

    col_tup = tuple(columns)
    for line in omni_file:                             #based on example:
        col_tup = line.split()                         #(key,value) = line.split()     #(key,value)=tuple
        for j,col in enumerate(columns):               
            omni_dict[col].append(col_tup[j])   #dict[key] = value 
    
    omni_df = pd.DataFrame(omni_dict)
    
    # Change datatypes to numeric
    omni_df = omni_df.apply(pd.to_numeric)

    # replace fill-values with NaNs
    # TODO: (JK) Allow for more flexibility for replacing fill values other than IMF
    omni_df.replace(to_replace=9999.99,value=np.nan,inplace = True)

    # Datetime column
    # TODO: (JK) Can think about moving this portion to 'convert_OMNI_EPOCH()' function
    # TODO: (JK) Find a way to optimize or save? (b/c this takes about 1min to run for 1yr-data)
    omni_df['Epoch'] = omni_df['Year'].astype(str)+'-'+omni_df['Day'].astype(str)+' '+omni_df['Hour'].astype(str)+':'+omni_df['Minute'].astype(str)
    omni_df['Epoch'] = omni_df['Epoch'].apply(pd.to_datetime,format='%Y-%j %H:%M')

    # Datetime index
    mag_df = omni_df.loc[:,~omni_df.columns.isin(['Year','Day','Hour','Minute'])]
    mag_df = mag_df.set_index('Epoch')
    return mag_df

def return_nans(mag_df, cols=None):
    """Determine the number of nans present

    Parameters
    ----------
    mag_df : pd.DataFrame
        Dataframe containing magnetometer data

    cols : list
        List of column names to process

    Returns
    -------
    [type]
        [description]
    """
    if cols is None:
        cols = mag_df.columns
    nan_data = {}
    for col in cols:
        # Count up the number of NaNs
        num_nans = sum(mag_df[col].isna())  
        nan_data[f'{col}_nnan'] = num_nans
    return nan_data

def interpolate_orbit(date, orbit, kind='linear'):
    """Function for interpolating an arbitrary time given the orbit of the spacecraft
    
    Parameters
    ----------
    date : pd.Timestamp
        Date when we want the postion of the instrument
    
    orbit : pd.DataFrame
        Dataframe holign information about the orbit of the instrument

    kind : str
        Type of interpolation to use. Default 'linear'
    
    Returns
    -------
    stats_data : dict
        `dict` containing the spacecraft position on the requested date
    """
    
    y = orbit.values
    interp_function = interp1d((orbit.index-orbit.index[0]).total_seconds().values, 
                    y, 
                    axis = 0, 
                    kind = kind)
    y_out = interp_function((date-orbit.index[0]).total_seconds())

    orbit_out = {}
    if y_out.ndim == 1:
        for i in range(0,y_out.shape[0]):
            orbit_out[orbit.keys()[i]] = y_out[i]
    else:    
        for i in range(0,y_out.shape[1]):
            orbit_out[orbit.keys()[i]] = y_out[:,i]
    
    return orbit_out