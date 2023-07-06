# Native Python imports (packages that come installed natively with Python)
import argparse
from collections import defaultdict
import glob
import datetime as dt
import logging
import sys
import os
import random

mod_dir = os.path.dirname(os.getcwd()).split('/')[:-1]
mod_dir.append('src')
data_module_directory = os.path.join('/', *mod_dir)
sys.path.append(data_module_directory)

# External packages
import hdbscan
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.style.use('default')
# plt.style.use('ggplot')
import matplotlib.dates as mdates
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

import wandb

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


parser = argparse.ArgumentParser()
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
    '-detrend_window',
    default=1800,
    help=(
        'Window size, in seconds, to use for detrending.'
        'Defaults to 1800 seconds.'
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
    '-smooth_window',
    default=2,
    help=(
        'Window size, in seconds, to use for smoothing.'
        'Defaults to [2].'
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


def plot_cluster(clusterer, distance_matrix):
    #plt.hist(clusterer.labels_,bins=np.arange(np.min(clusterer.labels_), np.max(clusterer.labels_)+2)-0.5)
    #plt.xlabel('Cluster Number')
    #plt.ylabel('Number of nodes')
    
    #clusterer.condensed_tree_.plot()
    #clusterer.single_linkage_tree_.plot()
    
    projection = TSNE(metric='precomputed').fit_transform(distance_matrix)

    (x,y) = projection.T[0], projection.T[1]

    color_palette = sns.color_palette('Paired', np.max(clusterer.labels_)+1)
    cluster_colors = [color_palette[x] if x >= 0
                    else (0.5, 0.5, 0.5)
                    for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                            zip(cluster_colors, clusterer.probabilities_)]
    fig, ax =plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=1)
    # for i, txt in enumerate(idx):
    #     ax.annotate(str(txt), (x[i], y[i]), fontsize=6)
    # plt.scatter(*projection.T, s=50, linewidth=0, c='k', alpha=0.25)
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    fout ='clusters.jpg'
    fig.savefig(fout, format='jpg', dpi=200, bbox_inches='tight')
    return fig, fout


def cluster(
    isax_pipe,
    component,
    node_level_depth, 
    node_sizes,
    min_node_size
):
    LOG.info('Running clustering of nodes...')
    idx = node_sizes[component].index[(node_sizes[component] > min_node_size)]
    # node_array = np.array(isax_pipe.nodes_at_level[component][node_level_depth])
    nodes_at_level = np.array(isax_pipe.nodes_at_level[component][node_level_depth])
    distance_matrix = np.zeros((len(nodes_at_level),len(nodes_at_level)))

    for i in range(0,len(nodes_at_level)-1):
        for j in range(i+1,len(nodes_at_level)):
            dis = nodes_at_level[i].get_min_max_distance(nodes_at_level[j])
            distance_matrix[i,j] = dis[3]
            distance_matrix[j,i] = dis[3]

    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=10, alpha=0.5)
    clusterer.fit(distance_matrix)
    return clusterer, distance_matrix
    
 
def plot_node(
    isax_pipe,
    component,
    node_level,
    ntimeseries,
    node_num,
    nsamples = 0,
    alpha = 0.5, 
    plot_bkt = False
):
    """Some code for plotting timeseries data at a node and 
    it's corresponding raw data

    Parameters
    ----------
    isax_pipe : bool
        [description]
    node_level : [type]
        [description]
    node_num : [type]
        [description]
    """
    isax_pipe.get_nodes_at_level(component=component, node_level=node_level)
    isax_pipe.get_node_info(component=component, node_level=node_level, node_num=node_num)
    # isax_pipe.get_node_info(node_level=node_level, node_num=node_num) 
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    data = isax_pipe.node_info[component][node_level][node_num]['annotation']
    # data.sort_index(ascending=True, inplace=True)
    #colors = plt.get_cmap('tab10')
    colors = colours(len(data)) 
    if nsamples !=0:
        if nsamples < data.shape[0]:
            rand_subset = np.random.randint(data.shape[0], size=nsamples)
            data = data.iloc[rand_subset].copy()
            data.sort_index(ascending=True, inplace=True)

    
    # Get breakpoints for cardinality
    bkpt = isax_pipe.sw_forest[component].forest[0].isax._card_to_bkpt(isax_pipe.min_cardinality)
    if plot_bkt:
        [ax[0].axhline(b, ls='--', c='r') for b in bkpt]
        [ax[0].text(x=310, y=b-0.5, s=str(i)) for i, b in enumerate(bkpt)]

    for j, (i, row) in enumerate(data.iterrows()):
        chunk = row['chunk_num']
        component = row['Component']
        node_name = row['Node Name']
        st, et = row['t0'], row['t1']
        # print(i, st, et)
        B = isax_pipe.interp_mag_seq[chunk][:, component]
        t = isax_pipe.interp_time_seq[chunk][:, 0].flatten()
        time_step = (t[1] - t[0])/np.timedelta64(1, 's')
        time_steps = [time_step * i for i in range(t.shape[0])]
        ax[0].plot(time_steps, B, c='k', alpha=alpha)
       
        delta = dt.timedelta(minutes=0)
        dt_raw = (isax_pipe.mag_df[st-delta:et+delta].index - isax_pipe.mag_df[st-delta:et+delta].index[0]).total_seconds()
        ax[1].plot(
            dt_raw,
            isax_pipe.mag_df[st-delta:et+delta][f'BRTN_{component}'],
            lw=0.5,
            c=colors[i],
            label=f"{(st-delta).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if j == 0:
            ax[1].axvspan(
                time_steps[0]+delta.seconds, delta.seconds+time_steps[-1],
                color='k',
                alpha=0.1,
                label='similar structure'
            )
    for a in ax:
        a.set_xlabel('Time Elapsed [seconds]')
        # a.set_ylim(-3, 5)
    ax[0].axhline(0, ls='-', c='r', lw=1)
    ax[1].axhline(0, ls='-', c='r', lw=1)
    ax[0].set_xlim(0, 330)
    ax[0].set_ylim(-5,5)
    ax[0].set_ylabel('Magnetic Field Component [nT]')
    leg = ax[1].legend(bbox_to_anchor=(1.04,1), loc="upper left", edgecolor='k', fontsize=9, markerscale=6)
    for line in leg.get_lines():
        line.set_linewidth(1.0)
    fig.suptitle(f'Node Name\n{node_name}, N={ntimeseries}')
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

def push_to_cloud(pdf_file, dirname):
    """Push the generate files to the bucket

    Parameters
    ----------
    pdf_file : [type]
        [description]
    """
    
    file_path = os.path.join(
        os.getcwd(),
        pdf_file
    )
    LOG.info(file_path)
    cmd = (
        f"gsutil -m cp -v {file_path} gs://isax-experiments-results/{dirname}"
    )
    os.system(cmd)
 

def run_experiment(
    input_file = None,
    start_date=dt.datetime(2018, 11, 21),
    stop_date=dt.datetime(2018, 12, 31),
    min_cardinality = 8,
    max_cardinality = 32,
    word_size = 10,
    threshold = 200,
    mu_x = 0.,
    std_x = 3.5,
    mu_y = 0.,
    std_y = 3.4,
    mu_z = 0.,
    std_z = 3.4,
    cadence=dt.timedelta(seconds=1),
    chunk_size=dt.timedelta(seconds=300),
    smooth_window=2,
    detrend_window=1800,
    node_level_depth = 16,
    min_node_size = 5

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
    """

    v = isax_vis.iSaxVisualizer()

    isax_pipe = isax_model.iSaxPipeline(
        orbit_fname = 'psp_orbit.csv',
        catalog_fname = 'psp_master_catalog_2018_2021_rads_norm.txt',
        threshold = threshold,
        word_size = word_size,
        min_cardinality = min_cardinality,
        max_cardinality = max_cardinality,
        mu_x = mu_x,
        std_x = std_x,
        mu_y = mu_y,
        std_y = std_y,
        mu_z = mu_z,
        std_z = std_z
    )   
    if input_file is None:
        catalog_cut = isax_pipe.catalog[start_date:stop_date]
        flist = list(catalog_cut['fname'].values)
        LOG.info(f'Found {len(flist)} between {start_date} {stop_date}')
    else:
        catalog_cut = pd.read_csv(input_file, header=0, index_col=0, parse_dates=True)
        flist = list(catalog_cut['fname'].values)
        LOG.info(f'Analyzing {len(flist)} between {catalog_cut.index[0]} {catalog_cut.index[-1]}')

    isax_pipe.run_pipeline(
        flist=flist,
        cadence=cadence,
        chunk_size=chunk_size,
        overlap = dt.timedelta(seconds=120),
        rads_norm=True,
        smooth=True,
        smooth_window=smooth_window,
        detrend=True,
        detrend_window=detrend_window,
        optimized=True
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

    x_mask = (node_sizes['x'] >= min_node_size)
    y_mask = (node_sizes['y'] >= min_node_size)
    z_mask = (node_sizes['z'] >= min_node_size)
    
    wandb.init(
        entity='solar-wind', 
        #name=component,
        project='CB_plots_week_7', 
        job_type='plot-isax-node',
        config=isax_pipe.input_parameters
    )
    parameter_file = isax_pipe.save(
        fname=f"isax_input_params_{chunk_size.seconds}.json",
        overwrite=False
    )
    dirname=f'chunk_size_{chunk_size.seconds:.0f}/'
    push_to_cloud(parameter_file, dirname=dirname)
    example_table = wandb.Table(columns=[
        "Component", "Number of Clusters", "Number of Nodes", "Cluster Image", "Node PDF"
        ]
    ) 
    for component, mask in zip(['x','y','z'], [x_mask, y_mask, z_mask]):
        pdf_file = f'isax_nodes_comp_{component}_chunk_size_{chunk_size.seconds:.0f}.pdf'
        pdf = PdfPages(pdf_file)        
        for node_num in tqdm(node_sizes[component].index[:20], desc=f'Plotting {component} nodes...'):
            fig = plot_node(
                isax_pipe,
                component,
                node_level = node_level_depth,
                ntimeseries = node_sizes[component].loc[node_num],
                node_num = node_num,
                nsamples = 5, 
                alpha = 0.5,
                plot_bkt = True
            )
            pdf.savefig(fig, bbox_inches='tight', dpi=200)
            plt.close(fig)
            # fig_list.append(fig)
        pdf.close()
        LOG.info('Pushing to the cloud!')
        push_to_cloud(pdf_file, dirname=dirname)
        hdbscan_clusters, distance_matrix = cluster(
            isax_pipe,
            component,
            node_level_depth, 
            node_sizes,
            min_node_size
        )

        
        
        cluster_fig, cluster_plot_file = plot_cluster(hdbscan_clusters, distance_matrix)
        push_to_cloud(cluster_plot_file, dirname=dirname)
        gsurl = f"https://storage.cloud.google.com/isax-experiments-results/{dirname}{pdf_file}"  
        
        example_table.add_data(
            component,
            np.max(hdbscan_clusters.labels_) + 1,
            node_sizes[component].shape[0],
            wandb.Image(cluster_fig),
            gsurl   
        )
        #example_table.add_column(name=f'Node Number, {component}', data=list(node_sizes[component].index[mask]))
        # example_table.add_column(name=f'Component {component}', data=fig_list)
 
    wandb.log({f"iSAX Experiment": example_table})
        
    wandb.finish()
   
    
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
    args['cadence'] = dt.timedelta(seconds=args['cadence'])
    args['chunk_size'] = dt.timedelta(seconds=args['chunk_size'])
    run_experiment(**args)