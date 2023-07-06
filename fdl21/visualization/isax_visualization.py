# Native Python imports (packages that come installed natively with Python)
from enum import unique
import datetime as dt
import sys
import os
import random
import logging

mod_dir = os.path.dirname(os.getcwd()).split('/')[:-1]
mod_dir.append('src')
data_module_directory = os.path.join('/', *mod_dir)
sys.path.append(data_module_directory)

# External packages
import numpy as np
import pandas as pd

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

import wandb

# # Native packages
import data.helper_funcs as hf
import model.isax_model as isax_model

class iSaxVisualizer():
    
    def __init__(self):
        """Visualization class for iSAX"""
        pass

    def colours(self, N, seed=None):
        """
        creates a list of random hex colours Parameters:
        -----------
        N: int
            the length of the array or list used for 
            plotting 
        """
        rng = np.random.default_rng(seed=seed)
        chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        col = ['#'+''.join(rng.choice(chars,6)) for i in range(N)]
        return col


    def linestyles(self, N):
        """
        creates a list of repeated linestyles:
        -----------
        N: int
            the lenght of the array or list used for 
            plotting 
        """
        ls_base = ['-', '--', ':', '-.']

        col = [ls_base[i%4] for i in range(N)]
        return col

    def optimal_time(self, t, max_t=None):
        """
        Finds the optimal interval to plot and returns the units and the converted time
        -----------
        t: seconds
            variable to convert to optimal units
        max_t: seconds
            if plotting raw data with a delta, this ensures that the conversion is the same for all windows
        
        Returns
        -------
        time: float
            converted time       
        unit: 
            unit of conversion        
        """
        if max_t is not None:
            max_t = np.max(t)

        if max_t < 120:
            return t, 'seconds'
        elif max_t < 3600*2:
            return t/60, 'minutes'
        elif max_t < 3600*48:
            return t/60/60, 'hours'
        else:
            return t/60/60/24, 'days'


    def plot_bkpts(self, 
                    ax, 
                    isax_pipe, 
                    component, 
                    cardinality=None, 
                    x_pos=None, 
                    add_text=False, 
                    **kwargs):
        """Function that plots the breakpoints on a given axis and the node's sequence and cardinality levels

        Parameters
        ----------
        ax : matplotlib.axes
            Axes to plot the breakpoints
        isax_pipe : isax_model.iSaxPipeline
            iSax pipeline object with isax definition
        component : str
            Magnetic field component to choose the right tree          
        cardinality : int
            cardinality of breakpoints
        x_pos : int
            the x position of the text (cardinality levels)
        add_text : bool
            If true, then the cardinality levels are printed
        **kwargs: Fill properties
            Properties passed to the fill_between function  
        """
        if cardinality is None:
            cardinality = isax_pipe.min_cardinality
        bkpt = isax_pipe.sw_forest[component].forest[0].isax._card_to_bkpt(cardinality)
        [ax.axhline(b, **kwargs) for b in bkpt]

        if add_text:
            x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
            if x_pos is None:
                x_pos =  ax.get_xlim()[1]
            [ax.text(x=x_pos + x_offset, y=b, s=str(i)+ '.' +str(cardinality), color='r', va='center', ha='left') for i, b in enumerate(bkpt)]
            


    def letter_bounds(self, 
                        ax, 
                        node, 
                        st=None, 
                        et=None, 
                        add_text=False, 
                        **kwargs):
        """Function that draws vertical boundaries between letters on a given curve plot

        Parameters
        ----------
        ax : matplotlib.axes
            Axes to plot the letter boundaries
        node : int
            The node that we're plotting the raw time series data from
        st : int
            The start time of the time interval
        et : int
            The end time of the time interval
        **kwargs: Fill properties
            Properties passed to the fill_between function  
        """

        if st is None:
            st =  ax.get_xlim()[0]

        if et is None:
            et =  ax.get_xlim()[1]

        # We calculate the word length from the node's short name
        word_length = len(list(map(float, node.short_name.split('_'))))
        # We calculate the time interval that each letter occupies
        interval = (et-st)/(word_length)
        # And we then graph each letter boundary according to the time interval calculated above
        [ax.axvline(b, **kwargs)for b in np.arange(st, et + interval, interval)]


        if add_text:
            interval = (et-st)/(word_length)
            x_pos = st+interval/2
            y_pos =  ax.get_ylim()[1]
            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0])*0.01
            seq = list(map(float, node.short_name.split('_')))
            word_length = len(list(map(float, node.short_name.split('_'))))

            for j in np.arange(len(seq)):
                ax.text(x=x_pos, y=y_pos+y_offset, s=str(seq[j]), va = 'bottom', ha='center', c='r')
                x_pos += interval


    def shaded_word(self,
                    ax, 
                    node, 
                    st=None, 
                    et=None, 
                    **kwargs):
        """Mini function that highlights the letters in an interpolated and detrended iSAX word plot

        Parameters
        ----------
        ax : matplotlib.axes
            Axes to plot the breakpoints
        node : node
            the node that we are visualizing
        st : int
            The start time of the time interval
        et : int
            The end time of the time interval
        **kwargs: Fill properties
            Properties passed to the fill_between function           
        """

        word_length = len(list(map(float, node.short_name.split('_'))))
        x_limits = ax.get_xlim()
        
        if st is None:
            st = x_limits[0]

        if et is None:
            et = x_limits[1]

        interval = (et-st)/(word_length)

        lower_bound, upper_bound, exp_value = node._do_bkpt()
        upper_bound = np.append(upper_bound, upper_bound[-1])
        lower_bound = np.append(lower_bound, lower_bound[-1])

        ax.fill_between(np.arange(st, et + interval, interval), upper_bound, lower_bound, step='post', **kwargs)


    def plot_node_curves(self, 
                        ax, 
                        nodes, 
                        isax_pipe, 
                        ntimeseries=5, 
                        rand_subset=None,
                        as_area=False, 
                        percentiles=[[10,90]], 
                        plot_median=False, 
                        max_t=None,
                        seed=42,
                        reload=False,
                        cache=False,
                        cache_folder=None,
                        instrument='psp',
                        **kwargs):
        """Some code for plotting timeseries data at a node and 
        it's corresponding raw data

        Parameters
        ----------
        ax : matplotlib.axes
            Axes to plot the breakpoints
        nodes: list
            List of nodes to plot
        isax_pipe : isax_model.iSaxPipeline
            iSax pipeline object with isax definition
        ntimeseries : int
            Number of time series to plot. Time series are randomly chosen. Overriden by rand_subset
            If None, plots all series.
        rand_subset : np.array
            subset of lines to plot. If provided, it overrides ntimeseries
        as_area: bool
            Flag that plots an area instead of the individual curves
        percentiles: list of lists
            list of lists of two numbers to be used as percentiles to plot an area
        plot_median: bool
            Flag that plots the median
        max_t: seconds
            if plotting raw data with a delta, this ensures that the conversion is the same for all windows
        seed: int
            Seed to use in random generator
        reload : bool
            forces the reload of the files
        cache : bool
            specifies whether a cache exists
        cache_folder : str
            provides location of the cache folder (where the cache is saved)
        instrument: str
            instrument to analyze
        **kwargs: Line2D or Fill properties
            Properties passed to the plotting of the time-series or the area
        Returns
        -------
        dates: pd.DataFrame
            Dataframe with the starting date for each time-series to be used as legend        
        node_names: list
            list with the node names that have been plotted
        end_time: float
            end time of the chunk window       
        """
        
        dates = pd.DataFrame()
        all_annotations = None
        B = None
        dates = None
        node_names = []
        rng = np.random.default_rng(seed=seed)
        n_sequences = 0

        # Get all annotations to minimize the amount of files that need to be loaded
        for node in nodes:
            node_names.append(node.short_name)
            n_sequences += node.nb_sequences
            if rand_subset is None or rand_subset.shape[0]>node.nb_sequences or len(nodes)>1:
                if ntimeseries is None:
                    ntimeseries = node.nb_sequences
                rand_subset = rng.choice(node.nb_sequences, np.min([ntimeseries, node.nb_sequences]), replace=False)
            
            data_node = node.get_annotations()
            data_node = data_node.iloc[rand_subset].copy()
            if all_annotations is None:
                all_annotations = data_node
            else:
                all_annotations = pd.concat([all_annotations, data_node])

        all_annotations.sort_values(by = 'File', ascending=True, inplace=True)
        unique_files = np.unique(all_annotations['File'])

        for file in unique_files:
            data = all_annotations.loc[all_annotations['File'] == file, :]  
            if reload:
                isax_pipe.get_data_for_plot(file, cache=cache, cache_folder=cache_folder, instrument=instrument)
                
            if as_area or plot_median:
                if B is None:
                    B = isax_pipe.interp_mag_seq[data['chunk_num'].values, :, data['Component'].values]
                else:
                    B = np.concatenate((B, isax_pipe.interp_mag_seq[data['chunk_num'].values, :, data['Component'].values]))

                t = pd.to_datetime(isax_pipe.interp_time_seq[0, :, 0].flatten())

                time_steps, units = self.optimal_time((t - t[0]).total_seconds(), max_t=max_t)        

            else:
                for j, (i, row) in enumerate(data.iterrows()):
                    chunk = row['chunk_num']
                    component = row['Component']
                    B = isax_pipe.interp_mag_seq[chunk][:, component]
                    t = pd.to_datetime(isax_pipe.interp_time_seq[chunk][:, 0].flatten())
                    time_steps, units = self.optimal_time((t - t[0]).total_seconds(), max_t=max_t)
                    ax.plot(time_steps, B, **kwargs)

                if dates is None:
                    dates = pd.DataFrame(index = data['t0'].values)
                else:
                    dates = dates.append(pd.DataFrame(index = data['t0'].values))

        if as_area:
            for percentile in percentiles:
                percentiles_plot = np.percentile(B, percentile, axis=0)
                ax.fill_between(time_steps, percentiles_plot[0,:], percentiles_plot[1,:], **kwargs)

        if plot_median and as_area:
            ax.plot(time_steps, np.nanmedian(B, axis=0), color='k')

        if plot_median and not as_area:
            ax.plot(time_steps, np.nanmedian(B, axis=0), **kwargs)


        ax.set_xlabel('time (' + units + ')')      

        return dates, node_names, time_steps[-1], n_sequences


    def plot_raw_data(self, 
                    ax, 
                    nodes, 
                    isax_pipe, 
                    delta=dt.timedelta(minutes=5), 
                    ntimeseries=5, 
                    rand_subset=None, 
                    shaded_region=True,
                    seed=42,
                    reload=False,
                    cache=False,
                    cache_folder=None,
                    instrument='psp',
                    **kwargs):
        """Some code for plotting timeseries data at a node and 
        its corresponding raw data

        Parameters
        ----------
        ax : matplotlib.axes
            Axes to plot the breakpoints
        nodes: list
            List of nodes to plot
        isax_pipe : isax_model.iSaxPipeline
            iSax pipeline object with isax definition
        ntimeseries : int
            Number of time series to plot. Time series are randomly chosen. Overriden by rand_subset
            If None, plots all series.
        rand_subset : np.array
            subset of lines to plot. If provided, it overrides ntimeseries
        shaded_region: bool
            plot the shaded area that highlights the region where isax is performed
        seed: int
            Seed to use in random generator
        reload : bool
            forces the reload of the files
        cache : bool
            specifies whether a cache exists
        cache_folder : str
            provides location of the cache folder (where the cache is saved)
        instrument : str
            which instrument we are analyzing
            instrument choices: 'psp' and 'wind'
        **kwargs: Line2D or Fill properties
            Properties passed to the plotting of the time-series or the area

        Returns
        -------
        dates: pd.DataFrame
            Dataframe with the starting date for each time-series to be used as legend        
        node_names: list
            list with the node names that have been plotted        
        """
        
        dates = pd.DataFrame()
        all_annotations = None
        B = None
        dates = None
        node_names = []
        rng = np.random.default_rng(seed=seed)

        # Get all annotations to minimize the amount of files that need to be loaded
        for node in nodes:
            node_names.append(node.short_name)
            if rand_subset is None or rand_subset.shape[0]>node.nb_sequences or len(nodes)>1:
                if ntimeseries is None:
                    ntimeseries = node.nb_sequences
                rand_subset = rng.choice(node.nb_sequences, np.min([ntimeseries, node.nb_sequences]), replace=False)
            
            data_node = node.get_annotations()
            data_node = data_node.iloc[rand_subset].copy()
            if all_annotations is None:
                all_annotations = data_node
            else:
                all_annotations = pd.concat([all_annotations, data_node])

        all_annotations.sort_values(by = 'File', ascending=True, inplace=True)
        unique_files = np.unique(all_annotations['File'])

        for file in unique_files:
            data = all_annotations.loc[all_annotations['File'] == file, :]  
            if reload:
                isax_pipe.get_data_for_plot(file, cache=cache, cache_folder=cache_folder, instrument=instrument)

            if dates is None:
                dates = pd.DataFrame(index = data['t0'].values)
            else:
                dates = dates.append(pd.DataFrame(index = data['t0'].values))            
        
            for j, (i, row) in enumerate(data.iterrows()):
                st, et = row['t0'], row['t1']
                
                cols = [val for val in isax_pipe.mag_df.columns if 'mag' not in val]
                cols = [val for val in cols if 'filename' not in val]                
                component = row['Component']

                try:
                    dt_raw = (isax_pipe.mag_df[st-delta:et+delta].index - isax_pipe.mag_df[st-delta:et+delta].index[0] - delta).total_seconds()
                    max_t = np.max(dt_raw)
                    dt_raw, units = self.optimal_time(dt_raw)
                    ax.plot(
                        dt_raw,
                        isax_pipe.mag_df[st-delta:et+delta][cols[component]],
                        **kwargs
                    )
                except:
                    LOG.exception("Chunk not available")

                if shaded_region:
                    chunk = row['chunk_num']
                    t = pd.to_datetime(isax_pipe.interp_time_seq[chunk][:, 0].flatten())
                    time_steps, units = self.optimal_time((t - t[0]).total_seconds())

                    ax.axvspan(
                        time_steps[0], time_steps[-1],
                        color='k',
                        alpha=0.1,
                        label='similar structure'
                    )

                    shaded_region = False

        ax.set_xlabel('time (' + units + ')') 

        return dates, node_names
