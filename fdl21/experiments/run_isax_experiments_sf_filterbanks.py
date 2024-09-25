# Native Python imports (packages that come installed natively with Python)
import argparse
from collections import defaultdict
import datetime as dt
import logging
import sys
import os
import random
from itertools import product
import cProfile
from pstats import Stats
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor as Pool


mod_dir = os.path.dirname(os.getcwd()).split('/')[:-1]
mod_dir.append('src')
data_module_directory = os.path.join('/', *mod_dir)
sys.path.append(data_module_directory)

# External packages
import hdbscan
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import dill as pickle
plt.style.use('default')
# plt.style.use('ggplot')
import matplotlib.dates as mdates
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import wandb
from anytree.exporter import DotExporter
from anytree import PreOrderIter

from copy import deepcopy

# Native packages
import fdl21.data.helper_funcs as hf
import fdl21.isax_model as isax_model
import fdl21.visualization.isax_visualization as isax_vis
import fdl21.data.build_filterbanks as fb
import fdl21.experiments.run_isax_experiments_sf as isax_experiment

wandb.login() 

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-filterbank_file',
    default=None,
    help='Filterbank pickle file path (saved from filterbank.py)'
)
parser.add_argument(
    '-input_file',
    default=None,
    help='Input file containing a list of files to process'
)

parser.add_argument(
    '-start_date',
    default='2018-11-21',
    help='Start date for interval. Defaults to 2018-11-21.'
)
parser.add_argument(
    '-stop_date',
    default='2018-12-31',
    help='Stop date for interval. Defaults to 2018-12-31.'
)
parser.add_argument(
    '-cadence',
    default=1,
    help=(
        'Final cadence of interpolated timeseries in seconds.'
        'Defaults to 1 second.'
    ),
    type=int
)
parser.add_argument(
    '-chunk_size',
    default=300,
    help=(
        'Duration, in seconds, of each chunk of the timeseries'
        'Defaults to 300 seconds.'
    ),
    type=int
)

parser.add_argument(
    '-max_cardinality',
    default=16,
    help='Maximum cardinality. Defaults to 16.',
    type=int
)
parser.add_argument(
    '-min_cardinality',
    default=8,
    help='Minimum cardinality. Defaults to 8.',
    type=int
)
parser.add_argument(
    '-node_level_depth',
    default=16,
    help=(
        'Deepest level to use when retrieving nodes.'
        'Defaults to 16.'
    ),
    type=int
)

parser.add_argument(
    '-threshold',
    default=200,
    help=(
        'Maximum number of timeseris a node can store.'
        'Defaults to 200.'
    ),
    type=int
)
parser.add_argument(
    '-word_size',
    default=10,
    help='SAX word length. Defaults to 10',
    type=int
)

parser.add_argument(
    '-overlap',
    default=0,
    help='Overlap used in chunking in seconds',
    type=int
)

parser.add_argument(
    '-cache',
    help='Flag that enables file cache.',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-cache_folder',
    default='/cache/',
    help='Folder to place cache file'
)

parser.add_argument(
    '-min_cluster_size',
    default=5,
    help='Min cluster size for HDBScan implementation',
    type=int
)

parser.add_argument(
    '-min_samples',
    default=5,
    help='Min sample size for HDBScan implementation',
    type=int
)

parser.add_argument(
    '-transliterate',
    help='Creating the transliterating files from the clustered chunks',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-instrument',
    default='psp',
    help='Instrument to analyze: psp or wind',
    type=str
)

parser.add_argument(
    '-cluster_selection_epsilon',
    default=None,
    help='Cluster selection epsilon for HDBScan as a percentile of the distance matrix distribution',
    type=float
)

parser.add_argument(
    '-n_processes',
    default=4,
    help='Number of processes for multipool',
    type=int
)

parser.add_argument(
    '-profiling',
    help='Runs profiler',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-plot_cluster',
    help='Plot clusters',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-parallel',
    help='Run parallel isax',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-recluster_iterations',
    default=0,
    help='Number of reclustering iterations',
    type=int
)

parser.add_argument(
    '-set_largest_cluster_to_noncluster',
    help='Set the largest cluster to Cluster -1 ("unclustered")',
    default=False,
    action='store_true'
)

parser.add_argument(
    '-save_model',
    help='Save isax_pipe object and clusters to pickle file',
    default=False,
    action='store_true'
)

def run_filterbanks(
    filterbank_file,
    input_file = None,
    start_date=dt.datetime(2018, 11, 21),
    stop_date=dt.datetime(2018, 12, 31),
    min_cardinality = 8,
    max_cardinality = 32,
    word_size = 10,
    threshold = 200,
    cadence=dt.timedelta(seconds=1),
    chunk_size=dt.timedelta(seconds=300),
    overlap = dt.timedelta(seconds=0),
    node_level_depth = 2,
    min_cluster_size = 5,
    min_samples = 5,
    cache=False,
    cache_folder= '/cache/', 
    transliterate = False,
    instrument='psp',
    cluster_selection_epsilon=None,
    n_processes=4,
    profiling=False,
    plot_cluster=False,
    parallel = False,
    recluster_iterations = 0,
    set_largest_cluster_to_noncluster = False,
    save_model = False
 ):
    """Run the iSAX experiment

    Parameters
    ----------
    input_file : [type], optional
        [description], by default None
    start_date : [type], optional
        [description], by default dt.datetime(2018, 11, 21)
    stop_date : [type], optional
        [description], by default dt.datetime(2018, 12, 31)
    min_cardinality : int, optional
        [description], by default 8
    max_cardinality : int, optional
        [description], by default 32
    word_size : int, optional
        [description], by default 10
    threshold : int, optional
        [description], by default 200
    mu_x : [type], optional
        [description], by default 0.
    std_x : float, optional
        [description], by default 3.5
    mu_y : [type], optional
        [description], by default 0.
    std_y : float, optional
        [description], by default 3.4
    mu_z : [type], optional
        [description], by default 0.
    std_z : float, optional
        [description], by default 3.4
    cadence : [type], optional
        [description], by default dt.timedelta(seconds=1)
    chunk_size : [type], optional
        [description], by default dt.timedelta(seconds=300)
    smooth_window : int, optional
        [description], by default 2
    detrend_window : int, optional
        [description], by default 1800
    overlap : int, optional
        [description], by default dt.timedelta(seconds=0)
    node_level_depth : int, optional
        [description], by default 16
    min_cluster_size : int, optional
        [description], by default 5
    min_samples : int, optional
        [description], by default 5
    cache : boolean, optional
        [description], by default False
    cache_folder : 'string', optional
        [description], by default '/cache/'
    transliterate : int, optional
        [description], by default False
    instrument: string
            instrument to analyze   
    cluster_selection_epsilon : [type], optional
        [description], by default None
    n_processes : int, optional
        [description], by default 4
    profiling : boolean, optional
        [description], by default False
    plot_cluster : boolean, optional
        [description], by default False
    parallel: boolean, optional
        [description], by default False
    recluster_iterations : int, optional
        Number of times to recluster Cluster -1 independently. 
        If 0, only does one initial cluster and never re-clusters cluster -1. 
        If greater than 0, then reclusters that many iterations. 
        If -1, "automatically" reclusters until reaches stability point 
        (doesn't create any new clusters) or until reaches maximum iteration count criteria.
        By default 0
    set_largest_cluster_to_noncluster : boolean, optional
        Not relevant if recluster_iterations=0. 
        Sometimes when clustering, an extremely large cluster that does not necessarily have any 
        meaningful patterns is created.
        Indicate whether or not to recluster the largest cluster along with cluster -1.
        By default False
    save_model : boolean, optional
        Indicate whether or not to save dictionaries of pipelines and clusters to pickle files 
        for each component.
        By default False
    """
    # Load saved filterbank (pickle) file
    isax_fb = fb.filterbank(restore_from_file=filterbank_file)

    # Run experiment for each filterbank
    for i,bank in enumerate(isax_fb.fb_matrix):
        # Extract edge frequencies for each filter
        if  isax_fb.DC or isax_fb.HF:                # if DC or HF filters exist...
            if isax_fb.DC and i == 0:                               # if DC filter, frequencies are first two
                bank_edge = isax_fb.edge_freq[:2]
            elif isax_fb.HF and i == isax_fb.fb_matrix.shape[0]:    # if HF filter, frequencies are last two
                bank_edge = np.array([isax_fb.edge_freq[-2],isax_fb.edge_freq[-1]])
            else:
                bank_edge = isax_fb.edge_freq[i-1:i+2]
        else:
            bank_edge = isax_fb.edge_freq[i:i+3]

        # Run experiment
        isax_experiment.run_experiment(
            input_file=input_file,
            start_date=start_date,
            stop_date=stop_date,
            min_cardinality=min_cardinality,
            max_cardinality=max_cardinality,
            word_size=word_size,
            threshold=threshold,
            cadence=cadence,
            preprocess='filter',
            chunk_size=chunk_size,
            edge_freq=bank_edge,
            frequency_weights=bank,
            frequency_spectrum=isax_fb.fftfreq,
            overlap=overlap,
            node_level_depth=node_level_depth,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cache=cache,
            cache_folder=cache_folder,
            transliterate=transliterate,
            instrument=instrument,
            cluster_selection_epsilon=cluster_selection_epsilon,
            n_processes=n_processes,
            profiling=profiling,
            plot_cluster=plot_cluster,
            parallel=parallel,
            recluster_iterations=recluster_iterations,
            set_largest_cluster_to_noncluster=set_largest_cluster_to_noncluster,
            save_model=save_model
        )


    
if __name__ == "__main__":
    args = vars(parser.parse_args())
    print(args['input_file'])
    args['start_date'] = dt.datetime.strptime(
        args['start_date'],
        '%Y-%m-%d'
    )
    args['stop_date'] = dt.datetime.strptime(
        args['stop_date'],
        '%Y-%m-%d'
    )

    run_prefix = f"CS{args['chunk_size']}_C{args['cadence']}_O{args['overlap']}_{args['instrument']}"
    profile_file = run_prefix + f"_WS{args['word_size']}_CA{args['min_cardinality']}_{args['max_cardinality']}_MCS{args['min_cluster_size']}"
    profile_file = profile_file + f"_MS{args['min_samples']}_T{args['threshold']}_NLD{args['node_level_depth']}_profile.txt"


    args['cadence'] = dt.timedelta(seconds=args['cadence'])
    args['chunk_size'] = dt.timedelta(seconds=args['chunk_size'])
    args['overlap'] = dt.timedelta(seconds=args['overlap'])
    
    if args['profiling']:
        pr = cProfile.Profile()
        pr.enable()
        run_filterbanks(**args)
        pr.disable()

        if not os.path.exists('profiles'):
            os.makedirs('profiles')

        with open('profiles/'+profile_file, 'w') as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('time')
            stats.dump_stats('.prof_stats')
            stats.print_stats()
    else:

        run_filterbanks(**args)