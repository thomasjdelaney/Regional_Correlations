import os, argparse, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.io import loadmat
from itertools import combinations, product
from scipy.stats import pearsonr

np.random.seed(1798) # setting seed
pd.set_option('max_rows',30) # setting display options for terminal display
# defining useful directories
proj_dir = os.path.join(os.environ['HOME'], 'Regional_Correlations')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
image_dir = os.path.join(proj_dir, 'images')

# loading useful functions
sys.path.append(py_dir)
import regionalCorrelations as rc

cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
cell_ids = cell_info[(cell_info.region=='thalamus')&(cell_info.group=='good')].index.values
trials_info = rc.getStimTimesIds(stim_info)
spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
trials_frame, num_bins, num_trials = rc.getTrialsFrame(trials_info, 0.005)

exp_frame = rc.getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, 0.005)
