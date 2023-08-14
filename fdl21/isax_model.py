"""
This class contains all the functionality required to generate
the iSAX tree and corresponding annotations. For each component
of the magnetic field, this code will generate a distinct iSAX tree.

"""

from collections import defaultdict
from collections.abc import Iterable
import datetime as dt
import glob
import json
import logging
import os
import sys
import time
from numpy.lib.function_base import iterable
from pandas.core.algorithms import isin

from scipy.linalg.basic import det
from tqdm import tqdm
import yaml
from pathlib import Path

# Add the ~/fdl-2021-solar-wind-repository/src directory to our path
# this will ensure we can import the necessary modules
_MODEL_DIR = os.path.dirname( os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_SRC_DIR)


# External packages
from anytree import RenderTree, PreOrderIter
from anytree.exporter import DotExporter, DictExporter
from anytree.importer import DictImporter
import dask
import numpy as np
import pandas as pd
from pyCFOFiSAX import ForestISAX
from tslearn.piecewise import PiecewiseAggregateApproximation


# Local packages
from fdl21.utils import time_chunking as tc
from fdl21.data import prototyping_metrics as pm
from fdl21.data import helper_funcs as hf


_PSP_MAG_DATA_DIR = '/sw-data/psp/mag_rtn/'
_WIND_MAG_DATA_DIR = '/sw-data/wind/mfi_h2/'
_OMNI_MAG_DATA_DIR = '/sw-data/nasaomnireader/'
_SRC_DATA_DIR = os.path.join(
    _SRC_DIR,
    'data',
)

_EXPONENTS_LIST = [2.15, 1.05, 1.05]

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


class iSaxPipeline(object):
    def __init__(
        self,
        start_date='2018-01-01',
        stop_date='2018-01-31',
        date_fmt = '%Y-%m-%d',
        catalog_fname=None,
        orbit_fname=None,
        min_cardinality=2,
        max_cardinality=16,
        mu_x=0.2,
        std_x=3.4,
        mu_y=0.2,
        std_y=3.4,
        mu_z=0.2,
        std_z=3.4,        
        threshold = 200,
        word_size = 10,
        instrument='psp'
    ):  
        

        # master catalog filename
        self._catalog_fname = os.path.join(
            _SRC_DATA_DIR,
            catalog_fname
        )

        # master catalog dataframe
        self._catalog = pd.read_csv(
            self._catalog_fname,
            index_col=0
        )
        if instrument == 'psp':
            fmt = '%Y%m%d%H'
        elif instrument == 'wind':
            fmt = '%Y%m%d'
        elif instrument == 'omni':
            fmt = '%Y%m%d'
        converter = lambda val: hf.fname_to_datetime(val, fmt=fmt)
        dates = self._catalog['fname'].apply(converter)
        self._catalog.index = pd.DatetimeIndex(dates, name='date')

        self._exponents_lists = _EXPONENTS_LIST
        self._current_file = None
        self._files_analyzed = []
        self._interp_time_seq = None
        self._interp_mag_seq = None
        self._mag_df = None
        self._max_cardinality = max_cardinality
        self._min_cardinality = min_cardinality
        self._msg_div = '-'*60
        
        self._mu = {'x': mu_x, 'y': mu_y, 'z': mu_z}
        self._std = {'x': std_x, 'y': std_y, 'z': std_z}
        self._nodes_at_level = {
            'x': {},'y': {}, 'z': {}, 'all': {}
        }
        self._node_info = {
            'x': {}, 'y': {}, 'z': {}, 'all':{}
        }

        # orbit filename
        if orbit_fname is not None:
            self._orbit_fname = os.path.join(
                _SRC_DATA_DIR,
                orbit_fname
            )

            # orbit dataframe
            self._orbit = pd.read_csv(
                self._orbit_fname,
                sep=",",
                comment ="#",
                index_col='EPOCH_yyyy-mm-ddThh:mm:ss.sssZ',
                parse_dates=['EPOCH_yyyy-mm-ddThh:mm:ss.sssZ'],
            )

        # Start time for analyzing
        self._start_date = dt.datetime.strptime(
            start_date,
            date_fmt
        )

        # Stop time for analyzing
        self._stop_date = dt.datetime.strptime(
            stop_date, 
            date_fmt
        )
        self.hist = None
        self._sw_forest = {'x': None, 'y': None, 'z': None, 'all': None}
        self._threshold = threshold
        self._ts = {'x': None, 'y': None, 'z': None, 'all': None}
        self._ts_loc = {'x': None, 'y': None, 'z': None, 'all': None}
        self._word_size = word_size
        self._input_parameters = {
            'threshold': self.threshold,
            'word_size': self.word_size,
            'min_cardinality': self.min_cardinality,
            'max_cardinality': self.max_cardinality,
            'files_analyzed': None,
            'rads_norm': None,
            'files_analyzed': None,
            'cadence': None,
            'chunk_size': None,
            'overlap': None,
            'smooth': None,
            'smooth_window': None,
            'detrend': None,
            'detrend_window': None,
            'optimized': None          
        }

    @property
    def catalog(self):
        """Attribute to store the PSP data catalog dataframe"""
        return self._catalog

    @property
    def current_file(self):
        """Attribute to store the current file we are analyzing"""
        return self._current_file

    @current_file.setter
    def current_file(self, value):
        self._current_file = value
    
    @property
    def interp_mag_seq(self):
        """Attribute for storing the interpolated, chunked magnetic field data"""
        return self._interp_mag_seq

    @interp_mag_seq.setter
    def interp_mag_seq(self, value):
        self._interp_mag_seq = value

    @property
    def interp_time_seq(self):
        """Attribute for storing the interpolated, chunked times"""
        return self._interp_time_seq

    @interp_time_seq.setter
    def interp_time_seq(self, value):
        self._interp_time_seq = value

    @property
    def input_parameters(self):
        return self._input_parameters

    @input_parameters.setter
    def input_parameters(self, value):
        self._input_parameters = value

    @property
    def mag_df(self):
        """pandas.DataFrame containing magnetic field data"""
        return self._mag_df

    @mag_df.setter
    def mag_df(self, value):
        self._mag_df = value
    
    @property
    def max_cardinality(self):
        """Maximum cardinality of SAX word"""
        return self._max_cardinality

    @max_cardinality.setter
    def max_cardinality(self, value):
        self._max_cardinality = value

    @property
    def min_cardinality(self):
        """Minimum cardinality of SAX word"""
        return self._min_cardinality

    @min_cardinality.setter
    def min_cardinality(self, value):
        self._min_cardinality = value

    @property
    def mu(self):
        """The mean of the entire dataset."""
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def nodes_at_level(self):
        """The nodes at a given level"""
        return self._nodes_at_level

    @nodes_at_level.setter
    def nodes_at_level(self, value):
        self._nodes_at_level = value

    @property
    def node_info(self):
        """Information generated for a given node"""
        return self._node_info

    @node_info.setter
    def node_info(self, value):
        self._node_info = value
    
    @property
    def std(self):
        """The standard deviation of the entire dataset."""
        return self._std 

    @std.setter
    def std(self, value):
        self._std = value

    @property
    def orbit(self):
        """The PSP orbit data"""
        return self._orbit

    @property
    def sw_forest(self):
        """The iSAX Forest tree object"""
        return self._sw_forest

    @sw_forest.setter
    def sw_forest(self, value):
        self._sw_forest = value

    @property
    def threshold(self):
        """The maximum number of timeseries held at a terminal node."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @property
    def ts(self):
        """Sequence of magnetic field data formatted for input to iSAX"""
        return self._ts
    
    @ts.setter
    def ts(self, value):
        self._ts = value

    @property
    def ts_loc(self):
        """Annotations of the sequences of magnetic field data"""
        return self._ts_loc
    
    @ts_loc.setter
    def ts_loc(self, value):
        self._ts_loc = value  

    @property
    def word_size(self):
        """The length of each SAX word"""
        return self._word_size

    @word_size.setter
    def word_size(self, value):
        self._word_size = value 

    def __str__(self):
        """Print a summary of the number info
        """
        num_nodes_analyzed = sum(
            [
                len(self.node_info[key].keys()) for key in self.node_info.keys()
            ]
        )
        node_msg = ''
        for node_key in self.node_info.keys():
            node_msg += f'Node level: {node_key}\n  Node Numbers:\n'
            for node_num in self.node_info[node_key].keys():
                node_msg += f"  {node_num}\n"
            node_msg += f"{'-'*60}\n"
        msg = (
            f'Number of Nodes Analyzed: {num_nodes_analyzed}\n'
            f'{node_msg}'
        )
        return msg

    def get_time_chunks(
        self, 
        mag_df,
        cols,
        cadence=dt.timedelta(seconds=1),
        chunk_size=dt.timedelta(seconds=300),
        overlap = dt.timedelta(seconds=0),
        smooth=True,
        smooth_window=60,
        detrend=True,
        detrend_window=1800,
        optimized=False,
        verbose=True
    ):
        """Run the time chunking step


        Parameters
        ----------
        mag_df : pandas.DataFrame
            DataFrame containing all the magnetic field data
        cols : list
            Columns of the pandas.DataFrame that should be chunked
        cadence : dt.timedelta
            The final cadence of the interpolated timeseries, default is 1 seconds
        chunk_size : dt.timedelta
            The duration of each chunk, default is 300 seconds
        smooth : bool
            Boolean flag for indicating whether or not to smooth the data, default is True
        smooth_window : int
            The size of the smoothing window in seconds, default is 60 seconds
        detrend : bool, optional
            Boolean flag for indicating whether or not to detrend the data, default is True
        detrend_window : int, optional
            The size of the smoothing window in seconds, default is 1800 seconds
        optimized : bool
            Flag that activates the optimized or non-optimized chunking function
        verbose : bool
            Flag that eables the printing of information as files are processed
 
        """
        st_t = time.time()
        if verbose:
            LOG.info('Smoothing, interpolating, and chunking the time series..')
        avg_sampling_rate, _, _ = pm.check_sampling_freq(self.mag_df)
        LOG.debug(f'Average sampling rate for current dataset: {avg_sampling_rate:0.3f}')
        #cols = [col for col in self.mag_df.columns if 'mag' not in col.lower()]        

        if optimized:
                self.interp_mag_seq, self.interp_time_seq, self.chunk_filelist = tc.time_chunking_optimized(
                    mag_df=mag_df,
                    cols=cols,
                    cadence=cadence,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    smooth=smooth,
                    smooth_window=smooth_window,
                    detrend=detrend,
                    detrend_window=detrend_window, 
                    avg_sampling_rate=avg_sampling_rate
            )
        else:
            # Run the time chunking to generate our interpolated time series
            self.interp_time_seq, self.interp_mag_seq, self.chunk_filelist = tc.time_chunking(
                mag_df=mag_df,
                cols=cols,
                cadence=cadence,
                chunk_size=chunk_size,
                smooth=smooth,
                smooth_window=smooth_window,
                detrend=detrend,
                detrend_window=detrend_window, 
                avg_sampling_rate=avg_sampling_rate
            )
        et_t = time.time()
        duration = et_t - st_t
        units = 'seconds'
        if duration > 60:
            units = 'minutes'
            duration /= 60
        if verbose:
            LOG.info(
                ('Finished the timechunking steps...\n'
                f"Total duration: {duration:0.3f} {units}\n{self._msg_div}")
            )

    def read_file(
        self, 
        fname,
        instrument='psp',
        rads_norm=True
    ):
        """Read in the dataset and format it for input to SAX tree

        Parameters
        ----------
        fname : str
            The filename we wish to process
        instrument: str
            Solar wind instrument to analyze
        rads_norm : bool
            Boolean flag for indicating whether or not to perform radial normalization
        """
        st_t = time.time()
        self._current_file = fname
        if instrument == 'psp':
            data_dir = _PSP_MAG_DATA_DIR
        elif instrument=='wind' :
            data_dir = _WIND_MAG_DATA_DIR
        elif instrument == 'omni':
            data_dir = _OMNI_MAG_DATA_DIR

        # Generate the full path to the file
        fname_full_path = os.path.join(
            data_dir,
            *fname.split('/') # this is required do to behavior of os.join
        )
        LOG.debug(f'Extracting data from:\n {fname_full_path}')
        
        if instrument == 'psp':
            mag_df = pm.read_PSP_dataset(
                fname=fname_full_path,
                orbit=self.orbit,
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
        
        mag_df['filename'] = [fname]*mag_df.shape[0]
        
        if self.mag_df is not None:   
            # if self.mag_df is not empty, concat mag_df with existing self.mag_df
            self.mag_df = pd.concat([self.mag_df, mag_df])            
        else:        
            # otherwise, self.mag_df is not built yet and this is first self.mag_df
            self.mag_df = mag_df
       
        et_t = time.time()
        duration = et_t - st_t
        units = 'seconds'
        if duration > 60:
            units = 'minutes'
            duration /= 60
        LOG.debug(
            ('Finished reading in the data...\n'
            f'Total duration: {duration:0.3f} {units}\n{self._msg_div}')
        )

    def _annotate(
        self, 
        fname,
        time_seq=None,
        mag_seq=None,
        chunk_num=None
    ):
        """Build the annotations for a single sequence of data

        Parameters
        ----------
        fname : str
            name of file associated with the sequence of data
        time_seq : np.array
            [description]
        mag_seq : np.array
            [description]
        chunk_num : int
            chunk number associated with the sequence of data

        Returns
        -------
        ts_x : list
            list of x-component of mag_seq (Bx)
        
        ts_y : list
            list of y-component of mag_seq (By)

        ts_z : list
            list of z-component of mag_seq (Bz)
        
        ts_x_loc : dict
            dictionary of annotations for Bx of mag_seq
        
        ts_y_loc : dict
            dictionary of annotations for By of mag_seq
        
        ts_z_loc : dict
            dictionary of annotations for Bz of mag_seq
        
        """
        
        size = mag_seq.shape[0]

        # Get the x-component data and annotations
        ts_x = list(mag_seq[:, 0].reshape(-1, size))
        x_size = len(ts_x)
        # Save annotations for Bx in dictionary
        ts_loc_x = {
            'File': [fname] * x_size,
            'Component':[0] * x_size,
            't0': list(time_seq.reshape(-1, size)[:, 0]),
            't1': list(time_seq.reshape(-1, size)[:, -1]),
            'chunk_num': [chunk_num] * x_size
        }

        # Get the y-component data and annotations
        ts_y = list(mag_seq[:, 1].reshape(-1, size))
        y_size = len(ts_y)
        ts_loc_y = {
            'File':[fname] * y_size,
            'Component':[1] * y_size,
            't0': list(time_seq.reshape(-1, size)[:, 0]),
            't1': list(time_seq.reshape(-1, size)[:, -1]),
            'chunk_num': [chunk_num] * y_size
        }

        # Get the Z-component data and annotations
        ts_z = list(mag_seq[:, 2].reshape(-1, size))
        z_size = len(ts_z)
        ts_loc_z = {
            'File':[fname] * z_size,
            'Component':[2] * z_size,
            't0': list(time_seq.reshape(-1, size)[:, 0]),
            't1': list(time_seq.reshape(-1, size)[:, -1]),
            'chunk_num': [chunk_num]*z_size
        }       
    
        return ts_x, ts_y, ts_z, ts_loc_x, ts_loc_y, ts_loc_z

    def build_annotations(self):
        """Build the annotations for all the sequences of chunked data

        This function loops through the time chunks and contructs
        the annotations used to identify the timeseries data stored in
        a given node. 
        """
        st_t = time.time()
        if isinstance(self.interp_mag_seq, Iterable):
            for i, (time_seq, mag_seq, fname) in enumerate(zip(self.interp_time_seq, self.interp_mag_seq, self.chunk_filelist)):                
                ts_x, ts_y, ts_z, ts_loc_x, ts_loc_y, ts_loc_z = self._annotate(
                    fname=fname,
                    time_seq = time_seq,
                    mag_seq = mag_seq,
                    chunk_num = i
                )
                if i == 0:
                    ts_x_final = [ts_x]
                    ts_y_final = [ts_y]
                    ts_z_final = [ts_z]

                    ts_x_loc_final = ts_loc_x
                    ts_y_loc_final = ts_loc_y           
                    ts_z_loc_final = ts_loc_z
                else:
                    # Add each new dataset as a list of lists.
                    # This will make sure the values are stacked as
                    # nested lists (i.e. 2D array with shape 
                    # (time chunk, 1, B field)
                    # The 1 cocrresponds to the [] we wrap around each list
                    ts_x_final += [ts_x]
                    ts_y_final += [ts_y]
                    ts_z_final += [ts_z]

                    # Update the dictionaries of annotations
                    for key in ts_x_loc_final.keys():
                        ts_x_loc_final[key] += ts_loc_x[key]
                        ts_y_loc_final[key] += ts_loc_y[key]
                        ts_z_loc_final[key] += ts_loc_z[key]
                
        # Convert the list of lists into a 2D numpy array
        # Use squeeze to remove the axis with a value of 1.
        # There will always be one because of how we combined the lists
        self.ts['x'] = np.array(ts_x_final).squeeze()
        self.ts['y'] = np.array(ts_y_final).squeeze()
        self.ts['z'] = np.array(ts_z_final).squeeze()

        # Convert the dictionaries of annotations to DataFrames
        self.ts_loc['x'] = pd.DataFrame(ts_x_loc_final)
        self.ts_loc['y'] = pd.DataFrame(ts_y_loc_final)
        self.ts_loc['z'] = pd.DataFrame(ts_z_loc_final)
        
        et_t = time.time()
        duration = et_t - st_t
        units = 'seconds'
        if duration > 60:
            units = 'minutes'
            duration /= 60
        LOG.info(f'Total duration: {duration:0.3f} {units}\n{self._msg_div}')
        
       
    def generate_nodes(self, component='x', save=False):
        """Generate the iSAX tree
        """
        
        st_t = time.time()
        if self.sw_forest[component] is None:
            self.sw_forest[component] = ForestISAX(
                size_word=self.word_size,
                threshold=self.threshold,
                data_ts=self.ts[component],
                base_cardinality=self.min_cardinality,
                max_card_alphabet=self.max_cardinality,
                number_tree=1,
                mu=self.mu[component],
                sig=self.std[component]
            )

        # Build the index
        self.sw_forest[component].index_data(
            self.ts[component], 
            annotation=self.ts_loc[component]
        )

        if save: 
            DotExporter(self.sw_forest.forest[0].root).to_picture("isax_testing.png")

        et_t = time.time()
        duration = et_t - st_t
        units = 'seconds'
        if duration > 60:
            units = 'minutes'
            duration /= 60
        LOG.info(f"Total duration: {duration:0.3f} {units}\n{self._msg_div}")

    def get_nodes_at_level(self, component=None, node_level=8):
        """Get all the nodes and terminal nodes at specific level

        Parameters
        ----------
        node_level : int
            The node level

        """
        self.nodes_at_level[component][node_level] = self.sw_forest[component].forest[0].get_nodes_of_level_or_terminal(node_level) 

    def get_node_info(self, component=None, node_level=8, node_num=10):
        """Get the node annotation and time series data for given level

        Parameters
        ----------
        node_level : int
            The node level
        node_num : int 
            The node number
        """
        
        annotations = self.nodes_at_level[component][node_level][node_num].get_annotations()
        sequences = self.nodes_at_level[component][node_level][node_num].get_sequences()
        # Check if we have already analyzed this node
        # this prevents overwriting the dictionary that is created
        if node_level in self.node_info.keys():
            self.node_info[component][node_level][node_num] = {
                'annotation': annotations,
                'sequence': sequences
            }
        else:
            self.node_info[component][node_level] = {
                node_num : {
                    'annotation': annotations,
                    'sequence': sequences
                }
            }

    def save(self, fname=None, overwrite=False):
        """Method for saving the root node of the iSAX tree and related parameters
        """
        
        if fname is None:
            todays_date = dt.datetime.today().strftime('%Y_%m_%d')
            fname = f"isax_pipeline_run_{todays_date}.json"
        
        
        LOG.info(
            ('Saving input parameters to following output file,\n'
            f"{fname}\n{'-'*60}")
        )
        with open(fname, 'w') as fobj:
            json.dump(
                self.input_parameters,
                fobj,
                sort_keys=True,
                indent=4
            )
        return fname

        # dct = DictExporter().export(self.sw_forest.forest[0].root)
        # with open(f"{fout.replace('.json','.yaml')}", "w") as fobj:  # doctest: +SKIP
        #     yaml.dump(dct, fobj)

    def load_root(self, fname):
        """Attempting to load the root node"""
        with open(fname, "r") as file:  # doctest: +SKIP
            read_in_dct =  yaml.load(file, Loader=yaml.Loader)
            root = DictImporter().import_(read_in_dct)

        self.sw_forest = ForestISAX(
                size_word=self.word_size,
                threshold=self.threshold,
                data_ts=np.array([]),
                base_cardinality=self.min_cardinality,
                number_tree=1,
                mu=self.mu,
                sig=self.std
        )
        self.sw_forest.forest[0].root = root

    def restore_variables(self, file):

        variables = np.load(file, allow_pickle=True)
        self.ts = variables['ts'].ravel()[0]
        self.ts_loc = variables['ts_loc'].ravel()[0]
        self.mag_df = variables['mag_df'].ravel()[0]['mag']
        self.interp_time_seq = variables['interp_time_seq'].ravel()[0]['time_seq']
        self.interp_mag_seq = variables['interp_mag_seq'].ravel()[0]['mag_seq']
        self.chunk_filelist = variables['chunk_filelist'].ravel()[0]['chunk_filelist']

    def build_cache(
        self, 
        file,
        cadence=dt.timedelta(seconds=1),
        chunk_size=dt.timedelta(seconds=300),
        overlap=dt.timedelta(seconds=0),
        rads_norm=True,
        smooth=True,
        smooth_window=dt.timedelta(seconds=5),
        detrend=True,
        detrend_window=dt.timedelta(seconds=1800),
        optimized = True,
        hist_max=50,
        bin_width=0.1,
        cache_folder='/cache/',
        instrument='psp'      
    ):
        """Build new cache file or reload existing one to calculate PAA histogram for stand dev
        and mean calculation

        A cache file consists various information of the time sequence of data associated with the passed file.


        Parameters
        ----------
        file : str
            The filename we wish to process
        cadence : dt.timedelta
            The final cadence of the interpolated timeseries, default is 1 seconds
        chunk_size : dt.timedelta
            The duration of each chunk, default is 300 seconds
        overlap : [type]
            [description]
        rads_norm : bool, optional
            Boolean flag for controlling the normalization of the magnetic field 
            to account for the decay of the field strength with heliocentric distance
        smooth : bool
            Boolean flag for indicating whether or not to smooth the data, default is True
        smooth_window : int
            The size of the smoothing window in seconds, default is 60 seconds
        detrend : bool, optional
            Boolean flag for indicating whether or not to detrend the data, default is True
        detrend_window : int, optional
            The size of the smoothing window in seconds, default is 1800 seconds
        optimized: bool
            Use optimized version of the time_chunking algorithm
        hist_max : [type]
            [description]
        bin_width : [type]
            [description]
        cache_folder : [type]
            [description]
        instrument: string
            instrument to analyze


        Returns
        ---------
        success : bool
            Flag that returns true if cache completed successfully

        """


        try:

            self._paa = PiecewiseAggregateApproximation(self.word_size)
            self.bins = np.arange(-hist_max,hist_max+2*bin_width,bin_width)-bin_width/2
            
            if self.hist is None:
                self.hist = {'x': np.zeros((self.bins.shape[0]-1)),      
                            'y': np.zeros((self.bins.shape[0]-1)),      
                            'z': np.zeros((self.bins.shape[0]-1))}

            # update the dictionary with out input parameters
            self.input_parameters['rads_norm'] = rads_norm
            # self.input_parameters['files_analyzed'] = flist
            self.input_parameters['cadence'] = cadence.seconds
            self.input_parameters['chunk_size'] = chunk_size.seconds
            self.input_parameters['overlap'] = overlap.seconds
            self.input_parameters['smooth'] = smooth
            self.input_parameters['smooth_window'] = smooth_window.seconds
            self.input_parameters['detrend'] = detrend
            self.input_parameters['detrend_window'] = detrend_window.seconds
            self.input_parameters['optimized'] = optimized

            # Check if the cache file already exists
            output_file = cache_folder + file.split('.')[0] + '.npz'
            cache_file = Path(output_file)
            if cache_file.is_file():
                LOG.info('Loading cached file...')
                self.restore_variables(output_file)
            else:
                LOG.info('Creating cached file...')
                self.read_file(
                    fname = file,
                    rads_norm=rads_norm,
                    instrument=instrument
                )

                cols = [val for val in self.mag_df.columns if 'mag' not in val]
                cols = [val for val in cols if 'filename' not in val]
                #cols = [val for val in self.mag_df.columns if 'mag' not in val]
                self.get_time_chunks(
                    mag_df = self.mag_df,
                    cols = cols,
                    cadence = cadence,
                    chunk_size = chunk_size,
                    overlap = overlap,
                    smooth = smooth,
                    smooth_window = smooth_window,
                    detrend = detrend,
                    detrend_window = detrend_window,
                    optimized = optimized
                )

                self.build_annotations()
                if not os.path.exists(cache_folder + '/' + file.split('/')[1] + '/'):
                    os.makedirs(cache_folder + '/' + file.split('/')[1] + '/')  

                np.savez(cache_folder + file.split('.')[0] + '.npz', 
                        ts = self.ts, 
                        ts_loc = self.ts_loc,
                        mag_df = {'mag': self.mag_df},
                        interp_time_seq = {'time_seq': self.interp_time_seq},
                        interp_mag_seq = {'mag_seq': self.interp_mag_seq},
                        chunk_filelist = {'chunk_filelist': self.chunk_filelist})            

            # Calculate Histogram
            bins = self.bins
            for component in ['x', 'y', 'z']:
                ts_comp = self.ts[component]
                # reshape component time series for PAA transformation
                ts_comp_reshape = ts_comp.reshape(ts_comp.shape + (1,))
                # PAA transformation
                fit_paa = self._paa.fit_transform(ts_comp_reshape)

                hist, _ = np.histogram(fit_paa.reshape(-1), bins=bins.astype('float'))
                self.hist[component] += hist

            return True

        except:
            return False


    def run_pipeline(
        self, 
        flist,
        cadence=dt.timedelta(seconds=1),
        chunk_size=dt.timedelta(seconds=300),
        overlap=dt.timedelta(seconds=0),
        rads_norm=True,
        smooth=True,
        smooth_window=dt.timedelta(seconds=5),
        detrend=True,
        detrend_window=dt.timedelta(seconds=1800),
        optimized = False, 
        parallel = True,
        cache=False,
        cache_folder='/cache/',
        instrument='psp'
    ):
        """Run the full pipeline

        Parameters
        ----------
        flist : list
            list of files that you want to process
        cadence : dt.timedelta
            The final cadence of the interpolated timeseries, default is 1 seconds
        chunk_size : dt.timedelta
            The duration of each chunk, default is 300 seconds
        rads_norm : bool, optional
            Boolean flag for controlling the normalization of the magnetic field 
            to account for the decay of the field strength with heliocentric distance
        smooth : bool
            Boolean flag for indicating whether or not to smooth the data, default is True
        smooth_window : int
            The size of the smoothing window in seconds, default is 60 seconds
        detrend : bool, optional
            Boolean flag for indicating whether or not to detrend the data, default is True
        detrend_window : int, optional
            The size of the smoothing window in seconds, default is 1800 seconds
        optimized: bool
            Use optimized version of the time_chunking algorithm
        parallel: bool
            Run process in parallel (NOT WORKING)
        cache: bool
            Use cached versions of the file, if available
        cache_folder: string
            location of the cache folder
        instrument: string
            instrument to analyze            
        """
        st_t = time.time()

        if cache and isinstance(flist, list) and len(flist) == 1:
            # Check if the cache file already exists
            output_file = cache_folder + flist[0].split('.')[0] + '.npz'
            cache_file = Path(output_file)
            if cache_file.is_file():
                self.restore_variables(output_file)
                
            else:
                cache=False

        if not cache:
            if isinstance(flist, list):
                for f in tqdm(flist,desc='Reading Files'):
                    self.read_file(
                        fname = f,
                        rads_norm=rads_norm,
                        instrument=instrument
                    )
            # update the dictionary with out input parameters
            self.input_parameters['rads_norm'] = rads_norm
            self.input_parameters['files_analyzed'] = flist
            self.input_parameters['cadence'] = cadence.seconds
            self.input_parameters['chunk_size'] = chunk_size.seconds
            self.input_parameters['overlap'] = overlap.seconds
            self.input_parameters['smooth'] = smooth
            self.input_parameters['smooth_window'] = smooth_window.seconds
            self.input_parameters['detrend'] = detrend
            self.input_parameters['detrend_window'] = detrend_window.seconds
            self.input_parameters['optimized'] = optimized        

            cols = [val for val in self.mag_df.columns if 'mag' not in val]
            cols = [val for val in cols if 'filename' not in val]
            #cols = [val for val in self.mag_df.columns if 'mag' not in val]
            self.get_time_chunks(
                mag_df = self.mag_df,
                cols = cols,
                cadence = cadence,
                chunk_size = chunk_size,
                overlap = overlap,
                smooth = smooth,
                smooth_window = smooth_window,
                detrend = detrend,
                detrend_window = detrend_window,
                optimized = optimized
            )
            
            # build the annotations
            self.build_annotations()

        if parallel:
            computations = []
            # generate the nodes in the iSAX tree
            for component in ['x', 'y', 'z']:
                computations.append(
                    dask.delayed(self.generate_nodes)(save=False, component=component)
                )
            dask.compute(*computations, n_workers=os.cpu_count()//3)
        else:
            for component in ['x', 'y', 'z']:
                self.generate_nodes(save=False, component=component)


        et_t = time.time()
        duration = et_t - st_t
        units = 'seconds'
        if duration > 60:
            units = 'minutes'
            duration /= 60
        LOG.info(
            ("Finished processing all files...\n"
             f"Total duration: {duration:0.3f} {units}\n{self._msg_div}")
        )

    def get_data_for_plot(self, file, cache=False, cache_folder=None, instrument='psp'):
        """Load data from a particular file for plotting purposes

        Parameters
        ----------
        file : string
            File to reload and place in memory
        cache: bool
            Use cached versions of the file, if available
        cache_folder: string
            location of the cache folder
        instrument: string
            instrument to analyze
        """
        if cache:
            # Check if the cache file already exists
            output_file = cache_folder + file.split('.')[0] + '.npz'
            cache_file = Path(output_file)
            if cache_file.is_file():
                self.restore_variables(output_file)

            else:
                cache = False

        if not cache:
            self.mag_df = None
            self.read_file(
                            fname = file,
                            rads_norm = self.input_parameters['rads_norm'],
                            instrument=instrument
                            )

            # create the dictionary with out input parameters
            
            cols = [val for val in self.mag_df.columns if 'mag' not in val and 'filename' not in val]
            self.get_time_chunks(
                mag_df = self.mag_df,
                cols = cols,
                cadence = dt.timedelta(seconds=self.input_parameters['cadence']),
                chunk_size = dt.timedelta(seconds=self.input_parameters['chunk_size']),
                overlap = dt.timedelta(seconds=self.input_parameters['overlap']),
                smooth = self.input_parameters['smooth'],
                smooth_window = dt.timedelta(seconds=self.input_parameters['smooth_window']),
                detrend = self.input_parameters['detrend'],
                detrend_window = dt.timedelta(seconds=self.input_parameters['detrend_window']),
                optimized = self.input_parameters['optimized'],
                verbose=False
            )