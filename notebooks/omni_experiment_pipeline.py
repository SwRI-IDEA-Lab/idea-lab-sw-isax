import os
os.chdir('C:/Users/rokka/GH-repos/idea-lab-sw-isax')

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

#catalog_fname = 'data/omni_master_catalog_1995_2019.csv'


if __name__ == "__main__":
    # Run experiment
    isax_exp_sf.run_experiment(input_file= None,                   #if None, cut of catalog is used
                            start_date= dt.datetime(2018, 1, 1),  # start date of catalog cut
                            stop_date= dt.datetime(2018, 2, 2),   # end date of catalog cut
                            cadence=dt.timedelta(seconds=60),
                            cache=True,
                            instrument='omni')