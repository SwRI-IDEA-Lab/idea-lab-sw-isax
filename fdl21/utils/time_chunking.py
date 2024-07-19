import numpy as np
from datetime import timedelta
import logging
from scipy.linalg._misc import _datacopied

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


def datetime_range(start, end, delta):
    """Return an array of dates between start and end dates at a regular delta interval

    Function to create regular intervals of dates for interpolating solar wind data into
    a regular cadence

    Parameters
    ----------

    start : datetime
        the beginning of our time interval 

    end : datetime
        the end of our time interval

    delta : datetime.timedelta
        the amount of time in between each datetime entry in our new array

    """
    results = []
    curr = start
    while curr < end:
        curr += delta
        results.append(curr)
    return pd.Series(results)


def time_chunking(
    mag_df,
    cols,
    cadence = timedelta(seconds=300), 
    chunk_size = timedelta(seconds=3600), 
    start_time = None, 
    end_time = None,
    kind = 'linear',
    detrend=False,
    detrend_window=timedelta(seconds=1800),
    smooth=False,
    smooth_window=timedelta(seconds=30),
    optimized = False,
    avg_sampling_rate=None,
    min_datapoints=0.5,
    return_pandas=False
):

    """Return segments that are each our time duration of choice

    This function will return the segments that we want to observe as numpy arrays.
    These numpy arrays are reshaped from the original data such that they accommodate
    the difference cadences and perfectly align with their respective datetimes. 
    The start and end times define the time duration of the segments that we want to 
    analyze. This time duration is inclusive of the start and end times as well. 
    The data is interpolated into the regular cadence specified by the user. 

    Parameters
    ----------

    mag_df : pandas.DataFrame
        Dataframe with pd.DatetimeIndex that contains the magnetic field data output from read_PSP_dataset(). 

    cadence : datetime.timedelta
        Frequency of measurements

    chunk_size : datetime.timedelta
        your time duration of segments/chunks

    start_time : datetime
        Date, hour, minute, second to start extraction

    end_time : datetime
        Date, hour, minute, second to start extraction

    kind : str
        Setting for the 'kind' argument in function interp1. Default = 'linear'

    smooth : bool
        Boolean flag to perform smoothing before interpolation

    smooth_window : datetime.timedelta
        Size of the smoothing window

    detrend_window : datetime.timedelta
        Size of the detrend window        

    avg_sampling_rate : int
        Average sampling rate compute by prototyping metrics.

    min_datapoints : float
        Minimum number of datapoints you must have in a time period to perform the smoothing.

    return_pandas : bool
        If True, then return the final products as a list of pandas.DataFrames
        
    Returns
    -------
    interp_time_seq : np.array

        A numpy array of datetime objects
    
    interp_mgn_seq : np.array

        A numpy array of all three magnetic components
        
    """
    filename_df = mag_df['filename'] 
    mag_df=mag_df[cols]
    mag_df.sort_index(inplace=True)
    if detrend:
        LOG.debug('Detrending')
        smoothed = mag_df.rolling(detrend_window,
            center=True
        ).mean()
        # Subtract the detrend_window (e.g. 30 minutes or 1800s) to detrend
        mag_df = mag_df - smoothed

    if smooth:
        LOG.debug('Smoothing')
        mag_df = mag_df.rolling(smooth_window,
            center=True
        ).mean()
    
    if optimized:
        interp_mag_df = pd.DataFrame(
        index=pd.date_range(
            start=mag_df.index[0], end=mag_df.index[-1], freq=cadence
        )
        )
        # The interpolated and smoothed magnetic field
        interp_mag_df = (
            mag_df
            .reindex(mag_df.index.union(interp_mag_df.index))
            .interpolate(method='index', kind='linear')
            .reindex(interp_mag_df.index)
        )

        interp_mag_seq, interp_time_seq = sliding_window(
            df = interp_mag_df, 
            cols = cols,
            chunk_size = chunk_size,
            overlap = overlap,
            cadence = cadence
        )

        interp_time_seq = interp_time_seq.reshape((interp_time_seq.shape[0], interp_time_seq.shape[1], 1))

    else:
        # Reshape input into correct format for time chunking magic
        mgn_variable = mag_df[cols].values
        time_variable = mag_df.index.to_series()

        # removing nan values from data
        nan_remover = np.isfinite(np.sum(mgn_variable, axis = 1))   # returns where nans are present in the data
        mgn_variable = mgn_variable[nan_remover]    # returns mgn_variable devoid of nans 
        time_variable = time_variable[nan_remover]

        # Assigning start and end times of segement's time duration
        if start_time == None:
            start_time = time_variable.iloc[0]
        if end_time == None:
            end_time = time_variable.iloc[-1]

        interp_function = interp1d(
            (time_variable-start_time).apply(lambda x: x.total_seconds()),
            mgn_variable,
            axis = 0,
            kind = kind,
            fill_value = 'extrapolate'
        )    
        #files = filename_df[start_time: end_time] 
        #print(files)

        # Creating time interval from start time and end time and interpolating
        interp_time = datetime_range(start_time, end_time, cadence)
        interp_mgn = interp_function((interp_time-start_time).apply(lambda x: x.total_seconds()))

        npoints = int(np.round(chunk_size/cadence))

        interp_mgn_seq = interp_mgn[0:int(interp_mgn.shape[0]/npoints)*npoints].reshape((-1, npoints, 3))
        interp_time_seq = interp_time[0:int(interp_time.shape[0]/npoints)*npoints].values.reshape((-1, npoints, 1))
        
    chunk_filelist = []
    for t in interp_time_seq:
        #print(t[0], t[-1])
        chunk_st = t[0][0] 
        chunk_et = t[-1][0] 
        chunk_file = np.unique(filename_df[chunk_st: chunk_et].values)
        if chunk_file.shape[0]==2:
            chunk_file = chunk_file[0]
        chunk_filelist.append(chunk_file[0])   #taking the first file from the overlaping due to the 0.27s  


    if return_pandas:
        dfs = [
            pd.DataFrame(val, columns=cols, index=t.flatten()) 
            for t, val in zip(interp_time_seq, interp_mgn_seq)
        ]
        return dfs

    return interp_time_seq, interp_mgn_seq, chunk_filelist


def sliding_window(
    df,
    cols,
    chunk_size = timedelta(minutes=5),
    overlap = timedelta(minutes=0),
    cadence = timedelta(seconds=1)
):
    """a sliding window to create smaller timeseries
    
    Parameters:
    ----------
    df: pandas dataframe
        input timeseries
    bins: time window
        number of minutes (default= 5) to 
        take for the window function
    step: stride size
        decides the overlapp between the sliding
        window
        default is 2 minutes

    Returns
    -------
    chunk_list: list of dataframes
        a list containing the sliced dataframes
    """
    if overlap.seconds == 0:
        overlap = chunk_size

    df = df[cols]
    chunk_mag = []
    chunk_time = []
    start = df.index[0]
    stop = df.index[-1]
    while start <= (stop-chunk_size):
        data = df[start: start+chunk_size]
        # idx = (start <= df.index) & (df.index < start+bins)
        time_chunk = np.array(
            pd.date_range(
                start=data.index[0], end=data.index[-1], freq=cadence
            )
        )
        chunk_time.append(
            time_chunk
        )
        chunk_mag.append(data.values)
        start += overlap
    return np.array(chunk_mag), np.array(chunk_time)
