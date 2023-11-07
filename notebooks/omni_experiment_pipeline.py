import os
os.chdir('/home/jasminekobayashi/idea-lab-sw-isax')

import glob
import datetime as dt
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
from concurrent.futures import ProcessPoolExecutor as Pool
from tqdm import tqdm

# import local libraries
import fdl21.data.prototyping_metrics as pm
import fdl21.data.generate_catalog as gc
from fdl21 import isax_model
from fdl21.experiments import run_isax_experiments_sf as isax_exp_sf
import fdl21.visualization.isax_visualization as isax_vis

#catalog_fname = '/home/jasminekobayashi/idea-lab-sw-isax/data/omni_master_catalog_1994_2023.csv'


if __name__ == "__main__":
    # Run experiment
    isax_exp_sf.run_experiment(input_file= None,                   #if None, cut of catalog is used
                               word_size=3,
                               start_date= dt.datetime(2018, 1, 1),  # start date of catalog cut
                               stop_date= dt.datetime(2018, 2, 2),   # end date of catalog cut
                               cadence=dt.timedelta(seconds=60),
                            #    chunk_size=dt.timedelta(seconds=600),
                               smooth_window=dt.timedelta(seconds=1800),
                               cache=True,
                               cache_folder='/home/jasminekobayashi/isax_cache/',
                               instrument='omni')