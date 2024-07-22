"""
This module is used to generate a data catalog containing 
information about the sampling rates, magnetic field, and 
the quality of the data.'
"""

import argparse
from collections import defaultdict
import glob
import logging
import time

import dask
import numpy as np
from tqdm import tqdm
import pandas as pd
import fdl21.data.prototyping_metrics as pm


# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-start_year',
    help='Starting year for catalog generation.',
    type=int,
    default=2007
)

parser.add_argument(
    '-debug',
    help='Number of files to process while debugging',
    type=int,
    default=0
)

parser.add_argument(
    '-parallel',
    help='Run the file processing in parallel',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-stop_year',
    help='Stopping year for catalog generation (inclusive).',
    type=int,
    default=2008
)

parser.add_argument(
    '-data_path',
    help='Path to solar wind data.',
    type=str,
    default='/sw-data/wind/mfi_h2/'
)

parser.add_argument(
    '-instrument',
    help='Instrument to process.',
    type=str,
    default='wind'
)

parser.add_argument(
    '-orbit_file',
    help='Path to PSP orbit.',
    type=str,
    default=None
)

parser.add_argument(
    '-histogram',
    help='Flag that enables the calculation of datat histogram.',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-hist_max',
    help='Maximum value to use in the histogram.',
    type=int,
    default=1100
)

parser.add_argument(
    '-bin_width',
    help='Bin width to use in the histogram.',
    type=float,
    default=1
)

parser.add_argument(
    '-rads_norm',
    help='Flag that enables radius normalization.',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-exponents_list',
    help = 'List of exponents to normalize all fields',
    type = list,
    default = [2.15, 1.05, 1.05]    # the three exps are for each respective mgn direction
)

def generate_catalog(
    start_year=None, 
    stop_year=None, 
    data_path=None,
    debug=None,
    parallel=False,
    instrument=None,
    orbit_file=None,
    save_number=100,
    histogram=False,
    hist_max=None,
    bin_width=None,
    rads_norm=False,
    exponents_list=None
    ):
    """Generate a catalog of all the data between start_year and stop_year

    This function will create a master catalog containing that 
    describes the quality of the data contained in a given CDF file.
    The start and stop years define a date interval that is 
    inclusive at the starting year and exclusive at the 
    stopping year, i.e. [start_year, stop_year).

    Parameters
    ----------
    start_year : int
        Year to start processing

    stop_year : int
        Year to stop processing

    data_path : str
        Path to directory containing the data

    debug : int
        Number of files to process when debugging. 
        Facilitates processing only handful of files to expedite debugging. 

    parallel : bool
        Flag for indicating if the processing should be done in parallel

    instrument : str
        Solar wind instrument to process

    orbit_file : str
        Location of the csv with instrument's orbit

    save_number : int
        Number of files after which the program saves progress.  Default 100

    histogram : bool
        Flag that triggers the creation of the data histogram

    hist_max : float
        Maximum magnetic field to be included in histogram

    bin_width : float
        Bin width to use in the histogram

    rads_norm : bool
        Flag that triggers the normalization of the magnetic field according to the 
        distance to the Sun

    exponents_list : np.array
        Provides list of exponents for each mgn component to normalize the mgn field with respect 
        to the radial distance from the Sun
    
    """
    
    save_file = f"{instrument}_master_catalog_{start_year:0.0f}_{stop_year:0.0f}"
    if rads_norm:
        save_file = save_file + '_rads_norm'

    LOG.info(instrument)

    # Create variable to store histogram
    if histogram:
        bins = np.arange(-hist_max,hist_max+2*bin_width,bin_width)-bin_width/2
        hist = np.zeros((bins.shape[0]-1,3))
    
    # List of years is exclusive, so the stop_year is excluded
    # TODO: (JK) I think the stop_year is actually inclusive now with the '+1' part, but good to double check
    years_to_process = np.arange(start_year, stop_year+1, 1)

    # Loop through each year we want to analyze and retrieve the filenames
    flist = []
    for year in years_to_process:
        tmp_flist = glob.glob(f"{data_path}/{year:0.0f}/*cdf")
        # Sort the filenames by the integer given by int(YYYYMMDD)
        tmp_flist.sort(key=lambda val: int(val.split('/')[-1].split('_')[-2]))
        # Add the sorted files into the master file list
        flist += tmp_flist
    
    LOG.info(f"Found {len(flist):0.0f} files to process")

    orbit = None
    if orbit_file is not None:
        orbit = pd.read_csv(orbit_file, sep=",",comment ="#", index_col='EPOCH_yyyy-mm-ddThh:mm:ss.sssZ')
        orbit.index = pd.to_datetime(orbit.index, format='%Y-%m-%dT%H:%M:%S')

    if debug:
        LOG.debug(f'Only processing {debug:0.0f} files...')  
        flist = flist[:debug]
    
    st = time.time()
    # Experimenting with parallelization
    # TODO: Finish implementing parallelization
    if parallel:
        delayed_objects = [
            dask.delayed(analyze_file)(fname=fname, instrument=instrument, orbit=orbit) for fname in flist
        ]

        computed_results = dask.compute(*delayed_objects)
        LOG.info(computed_results)
    else:
        data_out = defaultdict(list)
        for n,fname in tqdm(enumerate(flist)):
            data_dict, hist_out = analyze_file(fname, instrument, orbit=orbit, histogram=histogram, 
                                  hist_max=hist_max, bin_width=bin_width, rads_norm=rads_norm, 
                                  exponents_list=exponents_list)

            # Add datapoints to cummulative histogram
            if histogram:
                if hist_out.shape[0]>1:
                    hist = hist + hist_out

            for key, val in data_dict.items():
                data_out[key].append(val)
            if n%save_number == 0:
                df = pd.DataFrame(data_out)
                df.to_csv(
                    save_file + '.csv',
                    header=True
                )
                if histogram:
                    np.savez(save_file + '.npz', bins=bins, hist=hist)

        df = pd.DataFrame(data_out)
        df.to_csv(
            save_file + '.csv',
            header=True
        )
        if histogram:
            np.savez(save_file + '.npz', bins=bins, hist=hist)
    et = time.time()
    duration = et - st
    LOG.info(f'Execution time: {duration:0.3f}')


def analyze_file(fname=None,
    instrument=None, 
    orbit=None,
    histogram=False,
    hist_max=None,
    bin_width=None,
    rads_norm=False,
    exponents_list=None
    ):
    """Convenience function to handle all the steps

    This function is used to apply all the steps to generate
    the required data for the final catalog. 

    Parameters
    ----------
    fname : str
        Filename we wish to process
    instrument : str
        Instrument to process

    orbit : pd.DataFrame
        Dataframe with instrument's orbit

    histogram : bool
        Flag that triggers the creation of the data histogram

    hist_max : float
        Maximum magnetic field to be included in histogram

    bin_width : float
        Bin width to use in the histogram
    
    rads_norm : bool
        Flag that triggers the normalization of the magnetic field according to the 
        distance to the Sun

    exponents_list : np.array
        Provides list of exponents for each mgn component to normalize the mgn field with respect 
        to the radial distance from the Sun

    Returns
    -------
    data_dict : dict
        Python `dict` containing all of the data we 

    hist: np.array
        Histogram of each of the three components for a given file

    """

    hist = np.array([np.nan])
    if histogram:
        bins = np.arange(-hist_max,hist_max+2*bin_width,bin_width)-bin_width/2
        hist = np.zeros((bins.shape[0]-1,3))
    data_dict = {}  
    data_dict['fname'] = '/'+'/'.join(fname.replace('\\','/').split('/')[-2:])
    
    # Read in the dataset
    if instrument == 'wind':
        try: 
            mag_df = pm.read_WIND_dataset(fname)
        except Exception as e:
            msg = f"{e}\n Script crashed at file: {fname}"
            LOG.error(msg)
            return {}, np.array([np.nan])

        cols = ['B_mag', 'BGSE_0', 'BGSE_1', 'BGSE_2']
    elif instrument == 'psp':
        try:
            mag_df = pm.read_PSP_dataset(fname, orbit=orbit, rads_norm=rads_norm, exponents_list=exponents_list)
        except Exception as e:
            msg = f"{e}\n Script crashed at file: {fname}"
            LOG.error(msg)
            return {}, np.array([np.nan])
        cols = ['B_mag', 'BRTN_0', 'BRTN_1', 'BRTN_2']
    
    elif instrument == 'omni':
        try:
            mag_df = pm.read_OMNI_dataset(fname)
        except Exception as e:
            msg = f"{e}\n Script crashed at file: {fname}"
            LOG.error(msg)
            return {}, np.array([np.nan])
        cols = ['B_mag','BX_GSE','BY_GSE','BZ_GSE']

    # Calculate histogram
    if histogram:
        for i in range(0,3):
            hist[:,i], _ = np.histogram(mag_df.iloc[:,i+1].values, bins=bins)

    # Check for NaNs
    nan_data = pm.return_nans(mag_df, cols=cols)
    # Compute stats on the B field data
    B_field_data = pm.compute_B_field_stats(
        mag_df=mag_df,
         cols=cols
    )
    # Compute stats on the rolling difference of B field data
    B_diff_data = pm.compute_B_diff_stats(
        mag_df=mag_df,
        cols=cols
    )
    # Compute information about the sampling rates
    avg_sampling_freq, sr_stats_data, bad = pm.check_sampling_freq(
        mag_df=mag_df,
        min_sep=15,
        verbose=True
    )
    # Create a list of the dictionaries we want to store
    data_to_store = [
        nan_data,
        B_field_data,
        B_diff_data, 
        sr_stats_data
    ]
    # Calculate the instrument orbit and add it to dictionary
    if orbit is not None:
        midpoint = mag_df.index[0]+(mag_df.index[-1]-mag_df.index[0])/2
        instrument_position = pm.interpolate_orbit(midpoint, orbit)
        data_to_store.append(instrument_position)

    # Combine the dictionaries into a single dictionary
    for tmp in data_to_store:
        data_dict.update(tmp)
    # Update the data quality flag
    data_dict['bad'] = bad
    return data_dict, hist


if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args) 
    LOG.info(f'Compiling catalog for: {args}')
    generate_catalog(**args)
