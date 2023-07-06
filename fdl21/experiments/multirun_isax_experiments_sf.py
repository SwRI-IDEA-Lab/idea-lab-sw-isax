# Native Python imports (packages that come installed natively with Python)
from collections import defaultdict
import datetime as dt
import logging
import sys
import os
import random
from itertools import product
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
from matplotlib.figure import Figure
# plt.ioff()

import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

import wandb
from anytree.exporter import DotExporter


# # Native packages
import data.helper_funcs as hf
import model.isax_model as isax_model
import visualization.isax_visualization as isax_vis

wandb.login() 

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

_RUN_NAME = 'multirun_test'
_DATE_TIME = str(dt.datetime.now().strftime('%Y%m%d%H%M'))
_DIRNAME = f'{_RUN_NAME}_{_DATE_TIME}'


_N_PROCESSES = 30
_CACHE = True
_TRANSLITERATE = True
_PLOT_NODES = False
_PLOT_TREES = False
_INSTRUMENT = 'wind'
_CACHE_FOLDER = '/cache/'
_INPUT_FILE = ['/home/andres_munoz_j/sw2021/notebooks/data/isax_experiments/wind_joint_perihelion_aphelion_week_8.csv']
_DATE_RANGE = [[dt.datetime.strptime('2018-11-21', '%Y-%m-%d'), dt.datetime.strptime('2018-12-31', '%Y-%m-%d')]]
_CADENCE = [1]
_CADENCE = [dt.timedelta(seconds=x) for x in _CADENCE]
_CHUNK_SIZE = [300]
_CHUNK_SIZE = [dt.timedelta(seconds=x) for x in _CHUNK_SIZE]
_DETREND_WINDOW = [1800]
_DETREND_WINDOW = [dt.timedelta(seconds=x) for x in _DETREND_WINDOW]
_SMOOTH_WINDOW = [2]
_SMOOTH_WINDOW = [dt.timedelta(seconds=x) for x in _SMOOTH_WINDOW]
_OVERLAP = [0]
_OVERLAP = [dt.timedelta(seconds=x) for x in _OVERLAP]
_WORD_SIZE = [6,8]
_MIN_CARDINALITY = [2]
_MAX_CARDINALITY = [16]
_THRESHOLD = [10]
_NODE_DEPTH = [10]
_MIN_CLUSTER_SIZE = [2,5,8]
_MIN_SAMPLES = [None, 1,4,7]
_CLUSTER_EPSILON = [None]


_CONFIG = {
        'instrument': _INSTRUMENT,
        'input files': _INPUT_FILE,
        'date ranges': _DATE_RANGE,
        'cadences': _CADENCE,
        'chunk sizes': _CHUNK_SIZE,
        'detrend windows': _DETREND_WINDOW,
        'smooth windows': _SMOOTH_WINDOW,
        'overlaps': _OVERLAP,
        'word sizes': _WORD_SIZE,
        'min cardinalities': _MIN_CARDINALITY,
        'max cardinalities': _MAX_CARDINALITY,
        'thresholds': _THRESHOLD,
        'node_depths': _NODE_DEPTH,
        'min cluster sizes': _MIN_CLUSTER_SIZE,
        'min samples': _MIN_SAMPLES,
        'cluster epsilons': _CLUSTER_EPSILON
        }

def plot_cluster(clusterer, distance_matrix, cluster_file='clusters.png'):
    
    projection = TSNE(metric='precomputed').fit_transform(distance_matrix)

    (x,y) = projection.T[0], projection.T[1]

    color_palette = sns.color_palette('Paired', np.max(clusterer.labels_)+1)
    cluster_colors = [color_palette[x] if x >= 0
                    else (0.5, 0.5, 0.5)
                    for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                            zip(cluster_colors, clusterer.probabilities_)]

    fig = Figure(figsize=(8, 8))
    gridspec = fig.add_gridspec(nrows=1, ncols=1)    
    ax = fig.add_subplot(gridspec[0,0])

    ax.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=1)
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    fig.savefig(cluster_file, format='png', dpi=200, bbox_inches='tight')
    return fig


def cluster_function(
    isax_pipe,
    component,
    node_level_depth,
    min_cluster_size=5,
    min_samples=5,
    cluster_selection_epsilon=None
):
    LOG.info('Running clustering of nodes...')
    nodes_at_level = np.array(isax_pipe.nodes_at_level[component][node_level_depth])
    distance_matrix = np.zeros((len(nodes_at_level),len(nodes_at_level)))

    for i in tqdm(range(0,len(nodes_at_level)-1), desc=f'Calculating distance matrix...'):
        for j in range(i+1,len(nodes_at_level)):
            dis = nodes_at_level[i].get_min_max_distance(nodes_at_level[j])
            distance_matrix[i,j] = dis[3]
            distance_matrix[j,i] = dis[3]

    if cluster_selection_epsilon is None:
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
    else:
        epsilon = np.percentile(distance_matrix.reshape(-1)[distance_matrix.reshape(-1)>0], cluster_selection_epsilon/2)
        LOG.info(f'Mean distance={np.median(distance_matrix.reshape(-1)[distance_matrix.reshape(-1)>0])}, cluster_selection_epsilon={epsilon} at {cluster_selection_epsilon} percentile')
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)

    LOG.info('Clustering')
    clusterer.fit(distance_matrix)
    return clusterer, distance_matrix
    

def plot_cluster_curves(cluster, 
                        clusterer, 
                        component, 
                        node_level_depth,
                        isax_pipe, 
                        colors=100,
                        percentiles = [[10,90], [20,80], [30,70], [40,60]],
                        max_t = 600,
                        cache=False,
                        cache_folder=None,
                        instrument='psp'
                        ):
    """Function to plot the curves and information associated with a cluster

    Parameters
    ----------
    cluster : int
        number of cluster to print
    clusterer : hdbscan clusterer
        hdbscan clustering of nodes
    component : str
        component to plot
    ntimeseries : datetime
        [description]
    colors : list
        randomly generated list of colors used to generate clusters
    max_t : int
        The maximum time in seconds -- default set to 600s
    """
                        
    node_index = (clusterer.labels_== cluster).nonzero()[0]
    plt.rcParams.update({'font.size': 15})

    fig = Figure(figsize=(13, 4), dpi=300)
    gridspec = fig.add_gridspec(nrows=1, ncols=3)    
    ax = np.array([fig.add_subplot(gridspec[0,0]), fig.add_subplot(gridspec[0,1]), fig.add_subplot(gridspec[0,2])])

    nodes = [isax_pipe.nodes_at_level[component][node_level_depth][i] for i in node_index]
 
    v = isax_vis.iSaxVisualizer()
    colours = v.colours(colors)
    linestyles = v.linestyles(colors)

    custom_cycler = plt.cycler(color=colours) + plt.cycler(ls=linestyles)

    ax[0].set_prop_cycle(custom_cycler)
    _, _, end_time, n_sequences = v.plot_node_curves(ax[0], nodes, isax_pipe, 
                                        reload=True, 
                                        cache=cache, 
                                        cache_folder=cache_folder, 
                                        as_area=True, 
                                        plot_median=True, 
                                        max_t=max_t, 
                                        alpha=0.2, 
                                        percentiles=percentiles,
                                        instrument=instrument, 
                                        color='b',
                                        ec='None')

    v.plot_bkpts(ax[0], isax_pipe, component, cardinality=8, x_pos=None, add_text=False, color='r', ls='--', alpha=0.5, lw=0.5)
    v.letter_bounds(ax[0], nodes[0], 0, end_time, color='k', ls=':', alpha=0.3)

    shaded_region = True
    legend_artists = []
    node_names = []
    for i,node in enumerate(nodes):
        color = custom_cycler.by_key()['color'][i%100]
        ls = custom_cycler.by_key()['linestyle'][i%100]

        l, = ax[1].plot([-1, -1], [0, 0], color=color, ls=ls)
        legend_artists.append(l)
        node_names.append(node.short_name)

        v.shaded_word(ax[1], node, st=0, et=end_time, alpha=0.2, color=color)
        v.plot_node_curves(ax[2], [node], isax_pipe, reload=True, as_area=False, plot_median=True, max_t=max_t,
                            alpha=0.5, color=color, ls=ls, cache=cache, 
                            cache_folder=cache_folder, instrument=instrument)

        if shaded_region:
            shaded_region = False

    v.plot_bkpts(ax[1], isax_pipe, component, cardinality=8, x_pos=None, add_text=True, color='r', ls='--', alpha=0.5, lw=0.5)
    v.plot_bkpts(ax[2], isax_pipe, component, cardinality=8, x_pos=None, add_text=False, color='r', ls='--', alpha=0.5, lw=0.5)

    ax[1].set_xlabel(ax[0].get_xlabel())

    ax[2].yaxis.set_label_position("right")
    ax[2].yaxis.tick_right()

    ax[0].set_ylabel('B (nT)')
    ax[2].set_ylabel('B (nT)')

    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    ax[1].legend(legend_artists, node_names, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False)

    fig.suptitle(f'Cluster {cluster}, N={n_sequences}', y=0.9, va='bottom')

    return fig

 
def plot_node(
    node,
    isax_pipe,
    component,
    ntimeseries = 5,
    colors = 100,
    max_t = 600,
    cache=False, 
    cache_folder=None,
    instrument='psp'
):
    """Some code for plotting timeseries data at a node and 
    it's corresponding raw data

    Parameters
    ----------
    isax_pipe : bool
        [description]
    component : str
        [description]
    ntimeseries : datetime
        [description]
    colors : list
        randomly generated list of colors used to generate clusters
    max_t : int
        The maximum time in seconds -- default set to 600s
    """
    plt.rcParams.update({'font.size': 15})

    fig = Figure(figsize=(8, 4), dpi=300)
    gridspec = fig.add_gridspec(nrows=1, ncols=2)    
    ax = np.array([fig.add_subplot(gridspec[0,0]), fig.add_subplot(gridspec[0,1])])

    v = isax_vis.iSaxVisualizer()

    colours = v.colours(colors)
    linestyles = v.linestyles(colors)
    custom_cycler = plt.cycler(color=colours) + plt.cycler(ls=linestyles)

    ax[0].set_prop_cycle(custom_cycler)

    v.plot_node_curves(ax[0], [node], isax_pipe, reload=True, ntimeseries=ntimeseries, as_area=True, plot_median=False, percentiles=[[0,100]],
                        cache=cache, cache_folder=cache_folder, instrument=instrument, max_t=max_t, color='k', alpha=0.2, edgecolor='None')
    _, _, end_time, n_sequences = v.plot_node_curves(ax[0], [node], isax_pipe, reload=True, ntimeseries=ntimeseries, as_area=False, max_t=max_t, cache=cache, 
                        cache_folder=cache_folder)
    v.plot_bkpts(ax[0], isax_pipe, component, cardinality=8, x_pos=None, add_text=True, color='r', ls='--', alpha=0.5, lw=0.5)
    v.letter_bounds(ax[0], node, 0, end_time, color='k', ls=':', alpha=0.3)

    ax[1].set_prop_cycle(None)
    ax[1].set_prop_cycle(custom_cycler)

    dates, node_names = v.plot_raw_data(ax[1], [node], isax_pipe, reload=True, ntimeseries=ntimeseries, cache=cache, 
                                        cache_folder=cache_folder, instrument=instrument)

    ax[1].legend(dates.index.strftime('%Y-%m-%d %H:%M'), bbox_to_anchor=(-0.1, -0.15), loc='upper center', ncol=3, frameon=False)

    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()

    ax[0].set_ylabel('B (nT)')
    ax[1].set_ylabel('B (nT)')

    fig.suptitle(f'{node.short_name}, N={node.nb_sequences}', y=0.9, va='bottom')
    
    return fig

def colours(N):
    """
    creates a list of random hex colours Parameters:
    -----------
    N: int
        the lenght of the array or list used for 
        plotting 
    """
    chars = '0123456789ABCDEF'
    col = ['#'+''.join(random.sample(chars,6)) for i in range(N)]
    return col

def push_to_cloud(pdf_file, dirname, relative_folder=None):
    """Push the generate files to the bucket

    Parameters
    ----------
    pdf_file : [type]
        [description]
    """

    if relative_folder is not None:
        file_path = os.path.join(
            os.getcwd(),
            relative_folder,
            pdf_file
        )
    else:
        file_path = os.path.join(
            os.getcwd(),
            pdf_file
        )
    LOG.info(file_path)
    cmd = (
        f"gsutil -m cp -v {file_path} gs://isax-experiments-results/{dirname}/{pdf_file}"
    )
    os.system(cmd)



def run_cache(
    input_file = None,
    date_range=[dt.datetime(2018, 11, 21), dt.datetime(2018, 12, 31)],
    cadence=dt.timedelta(seconds=1),
    chunk_size=dt.timedelta(seconds=300),
    detrend_window=dt.timedelta(seconds=1800),
    smooth_window=dt.timedelta(seconds=2),
    overlap=dt.timedelta(seconds=0)
 ):
    """Create the file cache

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
    node_level_depth : int, optional
        [description], by default 16
    min_node_size : int
        Minimum node size to include in clustering
    transliterate : int, optional
        [description], by default False
    instrument: string
            instrument to analyze        
    """

    base_folder = f'CS{chunk_size.seconds}_C{cadence.seconds}_SW{smooth_window.seconds}_DW{detrend_window.seconds}_O{overlap.seconds}_{_INSTRUMENT}'
    cache_folder =  _CACHE_FOLDER + base_folder + '/'
    
    if _INSTRUMENT == 'psp':
        catalog_fname = 'psp_master_catalog_2018_2021_rads_norm.txt' 
    elif _INSTRUMENT == 'wind':
        catalog_fname = 'wind_master_catalog_2006_2022.txt' 

    isax_pipe = isax_model.iSaxPipeline(
        orbit_fname = 'psp_orbit.csv',
        catalog_fname = catalog_fname,
        instrument=_INSTRUMENT,
    )

    if input_file is None:
        catalog_cut = isax_pipe.catalog[date_range[0]:date_range[1]]
        flist = list(catalog_cut['fname'].values)
        LOG.info(f'Found {len(flist)} between {date_range[0]} {date_range[1]}')
    else:
        catalog_cut = pd.read_csv(input_file, header=0, index_col=0, parse_dates=True)
        flist = list(catalog_cut['fname'].values)
        LOG.info(f'Analyzing {len(flist)} between {catalog_cut.index[0]} {catalog_cut.index[-1]}')

    for file in flist:
        isax_pipe.mag_df = None
        isax_pipe.build_cache(
            file=file,
            cadence=cadence,
            chunk_size=chunk_size,
            overlap = overlap,
            rads_norm=True,
            smooth=True,
            smooth_window=smooth_window,
            detrend=True,
            detrend_window=detrend_window,
            optimized=True,
            cache_folder=cache_folder,
            instrument=_INSTRUMENT
        )

def run_experiment(
    input_file = None,
    date_range = [dt.datetime(2018, 11, 21), dt.datetime(2018, 12, 31)],
    cadence = dt.timedelta(seconds=1),
    chunk_size = dt.timedelta(seconds=300),
    detrend_window = dt.timedelta(seconds=1800),
    smooth_window = dt.timedelta(seconds=2),
    overlap = dt.timedelta(seconds=0),
    word_size = 10,
    min_cardinality = 8,
    max_cardinality = 16,
    threshold = 20,
    node_level_depth = 4,
    min_cluster_size = 5,
    min_samples = 5,
    cluster_selection_epsilon = None
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
    node_level_depth : int, optional
        [description], by default 16
    min_node_size : int
        Minimum node size to include in clustering
    transliterate : int, optional
        [description], by default False
    instrument: string
            instrument to analyze        
    """

  
    try:

        if cluster_selection_epsilon is None:
            cse_text = 'NA'
        else:
            cse_text = str(int(cluster_selection_epsilon*10))

        cache_folder = f'CS{chunk_size.seconds}_C{cadence.seconds}_SW{smooth_window.seconds}_DW{detrend_window.seconds}_O{overlap.seconds}_{_INSTRUMENT}'
        pdf_file = cache_folder + f'_WS{word_size}_CA{min_cardinality}_{max_cardinality}_T{threshold}_NLD{node_level_depth}_MCS{min_cluster_size}_MS{min_samples}_CSE{cse_text}'
        cache_folder =  _CACHE_FOLDER + cache_folder + '/'

        wbrun = wandb.init(
            entity='solar-wind', 
            group=_DIRNAME,
            name=pdf_file,
            project='CB_week_8_futures_' + _INSTRUMENT, 
            job_type='plot-isax-node',
            config=_CONFIG
        )     
        
        if _INSTRUMENT == 'psp':
            catalog_fname = 'psp_master_catalog_2018_2021_rads_norm.txt' 
        elif _INSTRUMENT == 'wind':
            catalog_fname = 'wind_master_catalog_2006_2022.txt'

        isax_pipe = isax_model.iSaxPipeline(
            orbit_fname = 'psp_orbit.csv',
            catalog_fname = catalog_fname,
            threshold = threshold,
            word_size = word_size,
            min_cardinality = min_cardinality,
            max_cardinality = max_cardinality,
            instrument=_INSTRUMENT
        )

        if input_file is None:
            catalog_cut = isax_pipe.catalog[date_range[0]:date_range[1]]
            flist = list(catalog_cut['fname'].values)
            LOG.info(f'Found {len(flist)} between {date_range[0]} {date_range[1]}')
        else:
            catalog_cut = pd.read_csv(input_file, header=0, index_col=0, parse_dates=True)
            flist = list(catalog_cut['fname'].values)
            LOG.info(f'Analyzing {len(flist)} between {catalog_cut.index[0]} {catalog_cut.index[-1]}')

        if _CACHE:
            for file in tqdm(flist, desc=f'Creating file caches and histograms...'):
                isax_pipe.mag_df = None
                isax_pipe.build_cache(
                    file=file,
                    cadence=cadence,
                    chunk_size=chunk_size,
                    overlap = overlap,
                    rads_norm=True,
                    smooth=True,
                    smooth_window=smooth_window,
                    detrend=True,
                    detrend_window=detrend_window,
                    optimized=True,
                    cache_folder=cache_folder,
                    instrument=_INSTRUMENT
                )

            LOG.info('Recalculating mean and standard deviations.')
            bins = isax_pipe.bins
            delta = np.nanmedian(bins[1:]-bins[0:-1])
            centers = (bins[1:]+bins[0:-1])/2
            for component in ['x', 'y', 'z']:
                hist = isax_pipe.hist[component]
                mu = np.sum(centers*hist*delta)/np.sum(hist*delta)
                
                sig = np.sum(np.power(centers-mu, 2)*hist*delta)
                sig = sig/np.sum(hist*delta)
                sig = np.sqrt(sig)

                isax_pipe._mu[component] = mu
                isax_pipe._std[component] = sig            
                LOG.info(f'mu = {mu} and sig={sig} for ' + component + ' component')

        for file in flist:
            isax_pipe.mag_df = None
            isax_pipe.run_pipeline(
                flist=[file],
                cadence=cadence,
                chunk_size=chunk_size,
                overlap = overlap,
                rads_norm=True,
                smooth=True,
                smooth_window=smooth_window,
                detrend=True,
                detrend_window=detrend_window,
                optimized=True,
                cache_folder=cache_folder,
                cache=_CACHE,
                instrument=_INSTRUMENT
            )


        node_sizes = defaultdict(list)
        LOG.info('Getting nodes for files')
        for component in ['x', 'y', 'z']:
            isax_pipe.get_nodes_at_level(
                component=component,
                node_level=node_level_depth
            )
            for node in isax_pipe.nodes_at_level[component][node_level_depth]:
                node_sizes[component].append(node.get_annotations().shape[0])

        for component in ['x', 'y', 'z']:
            node_sizes[component] = pd.Series(node_sizes[component])
            node_sizes[component].sort_values(ascending=False, inplace=True)

        if not os.path.exists('runs'):
            os.makedirs('runs')

        parameter_file = isax_pipe.save(
            fname= 'runs/' + pdf_file + '.json',
            overwrite=True
        )
    
        push_to_cloud(parameter_file.split('/')[1], dirname=_DIRNAME, relative_folder='runs/')

        if _TRANSLITERATE:
            component_annotations = {'x': pd.DataFrame(),'y': pd.DataFrame(), 'z': pd.DataFrame()}
            transliteration_file = pdf_file + '_transliteration.csv'

        component_dict = {}
        for component in ['x','y','z']:

            ## Clustering
            hdbscan_clusters, distance_matrix = cluster_function(
                isax_pipe,
                component,
                node_level_depth,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon
            )        

            pdf_file_n = pdf_file + '_' + f'{component}' + '_nodes'  + '.pdf'
            if _PLOT_NODES: pdf_n = PdfPages('runs/' + pdf_file_n)   

            pdf_file_c = pdf_file + '_' + f'{component}' + '_clusters'  + '.pdf'
            pdf_c = PdfPages('runs/' + pdf_file_c)

            tree_file = pdf_file + '_' + f'{component}' + '_tree'  + '.png'
            cluster_file = pdf_file + '_' + f'{component}' + '_tsne'  + '.png'

            if _TRANSLITERATE: component_annotations[component] = pd.DataFrame()
            
            for cluster in np.arange(-1, np.max(hdbscan_clusters.labels_)+1):

                cluster_annotations=pd.DataFrame()

                fig_c = plot_cluster_curves(cluster, hdbscan_clusters, component, node_level_depth, isax_pipe, colors=100, max_t = 2*chunk_size.seconds,
                                            cache=_CACHE, cache_folder=cache_folder)
                pdf_c.savefig(fig_c, bbox_inches='tight', dpi=200)

                if _PLOT_NODES: pdf_n.savefig(fig_c, bbox_inches='tight', dpi=200)
                plt.close(fig_c)

                node_index = (hdbscan_clusters.labels_==cluster).nonzero()[0]
                nodes = [isax_pipe.nodes_at_level[component][node_level_depth][i] for i in node_index]

                if _TRANSLITERATE or _PLOT_NODES:

                    for node in tqdm(nodes, desc=f'Plotting or transliterating {component} nodes for cluster {cluster}...'):

                        if _TRANSLITERATE:
                            cluster_annotations = pd.concat([cluster_annotations, node.get_annotations()]) 
                        
                        if _PLOT_NODES:
                            fig_n = plot_node(
                                            node,
                                            isax_pipe,
                                            component,
                                            ntimeseries = 5,
                                            colors = 100,
                                            max_t = chunk_size.seconds,
                                            cache=_CACHE,
                                            cache_folder=cache_folder
                                            )
                        
                            pdf_n.savefig(fig_n, bbox_inches='tight', dpi=200)
                            plt.close(fig_n)

                if _TRANSLITERATE:
                    cluster_annotations[f'Letter {component}'] = str(cluster)+component
                    component_annotations[component]  = pd.concat([component_annotations[component], cluster_annotations])

            if _TRANSLITERATE:
                component_annotations[component] = component_annotations[component].set_index(['File', 'chunk_num']).sort_index()
            
            pdf_c.close()
            LOG.info('Pushing to the cloud!')
            push_to_cloud(pdf_file_c, dirname=_DIRNAME, relative_folder='runs/')
            
            if _PLOT_NODES: 
                pdf_n.close()
                push_to_cloud(pdf_file_n, dirname=_DIRNAME, relative_folder='runs/')
            
            component_dict[f'TSNE {component}'] = plot_cluster(hdbscan_clusters, distance_matrix, cluster_file='runs/' + cluster_file)
            push_to_cloud(cluster_file, dirname=_DIRNAME, relative_folder='runs/')
            component_dict[f'gsurl_c {component}'] = f"https://storage.cloud.google.com/isax-experiments-results/{_DIRNAME}/{pdf_file_c}"  
            component_dict[f'gsurl_n {component}'] = f"https://storage.cloud.google.com/isax-experiments-results/{_DIRNAME}/{pdf_file_n}"

            if _PLOT_TREES:
                DotExporter(isax_pipe.sw_forest[component].forest[0].root).to_picture('runs/' + tree_file)
                push_to_cloud(tree_file, dirname=_DIRNAME, relative_folder='runs/')
            component_dict[f'gsurl_t {component}'] = f"https://storage.cloud.google.com/isax-experiments-results/{_DIRNAME}/{tree_file}"
        
        bucket_link = f"https://storage.cloud.google.com/storage/browser/isax-experiments-results/{_DIRNAME}"

        example_table = wandb.Table(columns=[
                        "Cadence",
                        "Chunk Size",
                        "Detrend Window",
                        "Smooth_Window",
                        "Overlap",
                        "Word Size",
                        "Min Cardinality",
                        "Max Cardinality",
                        "Threshold",
                        "Node Depth",
                        "Min Samples",
                        "Min Cluster Size",
                        "Cluster Epsilon",
                        "Number of Clusters", 
                        "TSNE x", 
                        "TSNE y", 
                        "TSNE z", 
                        "Cluster PDF x", 
                        "Cluster PDF y", 
                        "Cluster PDF z", 
                        "Node PDF x",
                        "Node PDF y",
                        "Node PDF z",
                        "Tree x",
                        "Tree y",
                        "Tree z",
                        "Bucket Link"
                        ]
        )

        example_table.add_data(
                cadence.seconds,
                chunk_size.seconds,
                detrend_window.seconds,
                smooth_window.seconds,
                overlap.seconds,
                word_size,
                min_cardinality,
                max_cardinality,
                threshold,
                node_level_depth,
                min_samples,
                min_cluster_size,
                cluster_selection_epsilon,
                np.max(hdbscan_clusters.labels_) + 1,
                wandb.Image(component_dict['TSNE x']),
                wandb.Image(component_dict['TSNE y']),
                wandb.Image(component_dict['TSNE z']),
                component_dict['gsurl_c x'],   
                component_dict['gsurl_c y'],   
                component_dict['gsurl_c z'],
                component_dict['gsurl_n x'],
                component_dict['gsurl_n y'],
                component_dict['gsurl_n z'],
                component_dict['gsurl_t x'],
                component_dict['gsurl_t y'],
                component_dict['gsurl_t z'],
                bucket_link
        )
        wbrun.log({f"iSAX": example_table})
        wbrun.finish()

        plt.close(component_dict[f'TSNE x']) 
        plt.close(component_dict[f'TSNE y']) 
        plt.close(component_dict[f'TSNE z']) 
        
        if _TRANSLITERATE:
            transliteration = component_annotations['x'].merge(component_annotations['y'], how='outer', left_index=True, right_index=True, suffixes=(None, '_y'))
            transliteration = transliteration.merge(component_annotations['z'], how='outer', left_index=True, right_index=True, suffixes=(None, '_y'))
            transliteration.sort_values(['File', 'chunk_num'], inplace=True)
            transliteration = transliteration.loc[:,['t0', 't1', 'Letter x', 'Letter y', 'Letter z']]
            transliteration.to_csv('runs/'+ transliteration_file)
            push_to_cloud(transliteration_file, dirname=_DIRNAME, relative_folder='runs/')

    except:
        e = sys.exc_info()[0]
        print(e)

if __name__ == "__main__":

    if _CACHE:

        cache_list = list(product(_INPUT_FILE, 
                                    _DATE_RANGE,
                                    _CADENCE,
                                    _CHUNK_SIZE,
                                    _DETREND_WINDOW,
                                    _SMOOTH_WINDOW,
                                    _OVERLAP))
 
        (input_file, 
        date_range, 
        cadence, 
        chunk_size, 
        detrend_window, 
        smooth_window, 
        overlap) = map(list, zip(*cache_list))

        with Pool(max_workers=_N_PROCESSES) as pool:
            pool.map(run_cache, input_file, date_range, cadence, chunk_size, detrend_window, smooth_window, overlap)

        LOG.info('Done with cache')     

    experiment_list = list(product(
                                _INPUT_FILE,
                                _DATE_RANGE,
                                _CADENCE,
                                _CHUNK_SIZE,
                                _DETREND_WINDOW,
                                _SMOOTH_WINDOW,
                                _OVERLAP,
                                _WORD_SIZE,
                                _MIN_CARDINALITY,
                                _MAX_CARDINALITY,
                                _THRESHOLD,
                                _NODE_DEPTH,
                                _MIN_CLUSTER_SIZE,
                                _MIN_SAMPLES,
                                _CLUSTER_EPSILON
                                ))

    (input_file, 
    date_range, 
    cadence, 
    chunk_size, 
    detrend_window, 
    smooth_window, 
    overlap,
    word_size,
    min_cardinality,
    max_cardinality,
    threshold,
    node_depth,
    min_cluster_size,
    min_samples,
    cluster_epsilon) = map(list, zip(*experiment_list))
          

    with Pool(max_workers=_N_PROCESSES) as pool:
        results = pool.map(run_experiment, 
                        input_file, 
                        date_range, 
                        cadence, 
                        chunk_size, 
                        detrend_window, 
                        smooth_window, 
                        overlap,
                        word_size,
                        min_cardinality,
                        max_cardinality,
                        threshold,
                        node_depth,
                        min_cluster_size,
                        min_samples,
                        cluster_epsilon)

        try:
            for return_value in results:
                print(return_value)
        except Exception as e:
            print(e)                        

    LOG.info('Done with experiments')