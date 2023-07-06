"""This module contains all the helper functions that """
import datetime as dt
import numpy as np
import pandas as pd
from collections import defaultdict
import uuid
from tqdm import tqdm

##Helper functions
def get_time_chunk(
    df, 
    time=None, 
    window_size=dt.timedelta(minutes=30)
    ):
    """
    Divides the time series to chunks

    The splitting of time series is carried out by defining
    a central timestep and by using timedelta to select a 
    range.

    Parameters
    ----------
    df : pandas dataframe
        a pandas dataframe containing a single timeseries
    time : timestamp
        the central timestamp in our selected data chunk
        default is the first timestamp of the dataframe.
    window_size : datetime timedelta 
        the desired timedelta
        default is 30 minutes

    Returns
    -------
    window : pandas dataframe
        a sliced and shortened dataframe
    """

    if time is None:
        time = df.index[0]

    start_window = time - window_size
    if start_window < df.index[0]:
        start_window = df.index[0]

    stop_window = time + window_size
    if stop_window > df.index[-1]:
        stop_window = df.index[-1]

    window = df[start_window:stop_window]

    return window

def get_time_chunck_list(
    df,
    resample=False,
    freq='5T'
    ):
    """
    Divides the time series to equally long chuncks

    Parameters
    ----------
    df : pandas dataframe
        a pandas dataframe containing a single timeseries
        This can either be a LoadDataframe object,
        or a raw dataFrame of the data.
    
    resample : boolean
        Resample the data to 3 second cadence

    freq : string
        length of time separator

    Returns
    -------
    df_split : list of dataFrames
        original dataFrame split into equal time chuncks
    """

    try:
        data = df.mag_df # if data comes from LoadDataframe class
    except Exception as e:
        # LOG.exception(e) 
        try:
            data = df # if dataFrame is passed in directly
        except Exception as e:
            # LOG.exception(e) 
            pass

    if resample:
        data = get_resampled(data)

    df_split = [g for n, g in data.groupby(pd.Grouper(freq=freq))]

    return df_split



def get_resampled(df,binsize='3s'):
    """resamples the pandas dataframe
    
    Parameters
    ----------
    df : pandas dataframe
        timeseries data that will be resample
    binsize : time in mins or seconds
        default is 3s, which means the timeseries
        is resampled to a frequency of 1 sample/3s
    
    Returns
    -------
    df_resampled : pandas dataframe
        downsampled timeseries 
    """
    df_resampled = df.resample(binsize).mean()
    return df_resampled

def sliding_window(
    df,
    bins= dt.timedelta(minutes=5),
    step = dt.timedelta(minutes=2)):
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
    chunk_list = []
    start = df.index[0]
    stop = df.index[-1]
    while start <= (stop-bins+step):
        idx = (start <= df.index) & (df.index < start+bins)
        start += step
        chunk_list.append(df[idx])
        #print()
    return (chunk_list)

def fname_to_datetime(fname, fmt='%Y%m%d'):
    """function to get datatime object from filename

    Parameters
    ----------
    fname: string
        filename to be turned into datetime object

    Returns
    ----------
    dateobject: datetime object
        converted filename string into datetime object
    """
    date_str = fname.split('/')[-1].split('_')[-2]
    date_obj = dt.datetime.strptime(date_str, fmt)
    return date_obj
    

def check_for_discontinuity(
    df_labels,
    split_list,
    filename=None
    ):
    """ Function to check whether labelled discontinuty falls in a timechunck

    Parameters
    ----------

    df_labels : dataFrame
        dataFrame of labelled dataset with start/stop times of discontinuities
    
    split_list : list of dataFrames
        list of equally sized timechuncks

    filename : string
        name of the file that was split into df_split
    
    Returns
    -------
    df : DataFrame
        dataFrame matching timechunck information with labels
    """ 
    df_labels.index = pd.DatetimeIndex(((df_labels['End Time'] - df_labels['Start Time']) / 2).round('s') + df_labels['Start Time'], name='Mean Time')

    tmp_dict = defaultdict(list)

    for n in range(len(split_list)):
        start_time = split_list[n].index[0]
        stop_time = split_list[n].index[-1]

        cut = df_labels[start_time:stop_time]

        if cut.empty:
            is_discontinuity = 0
        else:
            is_discontinuity = 1

        if filename is not None:
            tmp_dict['filename'].append(filename)

        id_num = uuid.uuid1()

        tmp_dict['Start Time '].append(start_time)
        tmp_dict['Stop Time'].append(stop_time)
        tmp_dict['Number'].append(n)
        tmp_dict['chunck_length'].append(len(split_list[n]))
        tmp_dict['uniqe_id'].append(id_num)
        tmp_dict['is_discontinuity'].append(is_discontinuity)

    df = pd.DataFrame(tmp_dict)

    return df

    
