import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import datetime as dt
from scipy.io import loadmat
from itertools import product
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='For signal correlation vs time bin plots. ')
parser.add_argument('-g', '--group', help='The quality of sorting for randomly chosen_cells.', default='good', choices=['good', 'mua', 'unsorted'], type=str, nargs='*')
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

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

bin_widths = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
trials_info = rc.getStimTimesIds(stim_info)
for region in ['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1']:
    cell_ids = cell_info[(cell_info.region==args.region)&(cell_info.group==args.group)].index.values
    spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
    responding_pairs = rc.getRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, 30, is_strong=True)
    responding_cells = np.unique(responding_pairs)
    correlations = np.zeros(bin_widths.size)
    p_values = np.zeros(bin_widths.size)
    for i, bin_width in enumerate(bin_widths):
        exp_frame = rc.getExperimentFrame(responding_cells, trials_info, spike_time_dict, cell_info, 2.0)
        agg_frame = exp_frame[['cell_id', 'stim_id', 'num_spikes']].groupby(['cell_id', 'stim_id']).agg('mean')
        correlations[i], p_values[i] = pearsonr(agg_frame.loc[pair[0]], agg_frame.loc[pair[1]])
