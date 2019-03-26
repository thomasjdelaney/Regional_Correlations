import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.io import loadmat
from itertools import product, combinations
from scipy.stats import pearsonr

np.random.seed(1798) # setting seed
pd.set_option('max_rows',30) # setting display options for terminal display

# defining useful directories
proj_dir = os.path.join(os.environ['SPACE'], 'Regional_Correlations')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
image_dir = os.path.join(proj_dir, 'images')

# loading useful functions
sys.path.append(py_dir)
import regionalCorrelations as rc

def rollCorrCoef(first_trial, second_trial, roll):
    if roll == 0:
        corr_coef, p_value = pearsonr(first_trial, second_trial)
    elif roll > 0:
        corr_coef, p_value = pearsonr(first_trial[roll:], np.roll(second_trial, roll)[roll:])
    else:
        corr_coef, p_value = pearsonr(first_trial[:roll], np.roll(second_trial, roll)[:roll])
    return corr_coef, p_value

cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
cell_ids = cell_info[(cell_info.region=='thalamus')&(cell_info.group=='good')].index.values
trials_info = rc.getStimTimesIds(stim_info, 2)
num_trials = trials_info.shape[0]
spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
responding_pairs = rc.getRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, 5, is_strong=True)
responding_cells = np.unique(responding_pairs)
exp_frame = rc.getExperimentFrame(responding_cells, trials_info, spike_time_dict, cell_info, 0.001)
pair = responding_pairs[0]
first_response = exp_frame[exp_frame.cell_id == pair[0]]['num_spikes']
second_response = exp_frame[exp_frame.cell_id == pair[1]]['num_spikes']
num_responses = first_response.size
first_trials = first_response.values.reshape(num_trials, num_responses//num_trials)
second_trials = second_response.values.reshape(num_trials, num_responses//num_trials)
trial_combinations = combinations(np.arange(num_trials),2)
rolls = np.arange(-1000, 1001)
corr_coefs_by_offset = np.zeros([num_trials, rolls.size])
for i in np.arange(num_trials):
    first_trial = first_trials[i]
    second_trial = second_trials[i]
    corr_coefs_by_offset[i] = np.array([rollCorrCoef(first_trial, second_trial, r)[0] for r in rolls])
