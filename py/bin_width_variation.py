"""
For calculating the pairwise correlations between many neurons using different bin widths.
"""
import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.io import loadmat
from itertools import combinations
from scipy.stats.stats import pearsonr

parser = argparse.ArgumentParser(description='Calculate pairwise correlations between given choice of neurons.')
parser.add_argument('-n', '--number_of_pairs', help='The number of strongly responding pairs to use.', default=30, type=int)
parser.add_argument('-g', '--group', help='The quality of sorting for randomly chosen_cells.', default='good', choices=['good', 'mua', 'unsorted'], type=str, nargs='*')
parser.add_argument('-r', '--region', help='Filter the randomly chosen cells by region', default='thalamus', choices=['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1'], type=str) # only one region at a time here
parser.add_argument('-j', '--stim_id', help='A stim_id for use in the correlations vs bin length.', default=2, type=int)
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed) # setting seed
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

# loading stimulus info TODO: think about using the other stimuli
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))

# for a region, and a stimulus
# 1. get 30 randomly chosen strongly responding pairs
# 2. calculate the correlation coefficients for this pair, for all the different bin widths
# 3. plot corr vs bin width, save plot
# 4. append the frame to a csv

def getStronglyRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, num_pairs, strong_threshold=40.0):
    big_frame = rc.getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, 0.0)
    agg_frame = big_frame[['cell_id', 'num_spikes']].groupby('cell_id').agg({'num_spikes':'mean'}).sort_values('num_spikes', ascending=False)
    strongly_responding_cells = agg_frame[agg_frame >= 40.0].index.values
    all_strong_pairs = np.array(list(combinations(strongly_responding_cells, 2)))
    num_strong_pairs = all_strong_pairs.shape[0]
    randomly_chosen_pairs = all_strong_pairs[np.random.choice(np.arange(0,num_strong_pairs), num_pairs, replace=False),:]
    return randomly_chosen_pairs

cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
cell_ids = cell_info[(cell_info.region==args.region)&(cell_info.group==args.group)].index.values
trials_info = rc.getStimTimesIds(stim_info, args.stim_id)
spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
strong_pairs = getStronglyRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, args.number_of_pairs)
num_pairs = strong_pairs.shape[0]
strong_cells = np.unique(strong_pairs.flatten())
spike_time_dict = {k: spike_time_dict[k] for k in strong_cells}
bin_widths = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

bin_width = bin_widths[0]
strong_exp_frame = rc.getExperimentFrame(strong_cells, trials_info, spike_time_dict, cell_info, bin_width)
correlation_coefficients = np.zeros(num_pairs)
p_values = np.zeros(num_pairs)
for i, strong_pair in enumerate(strong_pairs):
    first_response = strong_exp_frame[strong_exp_frame.cell_id == strong_pair[0]]['num_spikes']
    second_response = strong_exp_frame[strong_exp_frame.cell_id == strong_pair[1]]['num_spikes']
    correlation_coefficients[i], p_values[i] = pearsonr(first_response, second_response)
width_frame = pd.DataFrame({'stim_id':np.repeat(args.stim_id, num_pairs), 'region':np.repeat(args.region, num_pairs), 'first_cell_id':strong_pairs[:,0], 'second_cell_id':strong_pairs[:,1], 'corr_coef':correlation_coefficients, 'p_value':p_values, 'bin_width':np.repeat(bin_width, num_pairs)})

def saveStronglyRespondingCorrelation(exp_frame, cell_info, bin_length, stim_id, region):
    strong_pair = getStronglyRespondingPair(exp_frame, cell_info, args.bin_length, stim_id, region)
    first_response = exp_frame[(exp_frame.cell_id == strong_pair[0])&(exp_frame.stim_id == stim_id)]['num_spikes']
    second_response = exp_frame[(exp_frame.cell_id == strong_pair[1])&(exp_frame.stim_id == stim_id)]['num_spikes']
    corr_coef, p_value = pearsonr(first_response, second_response)
    corr_info_frame = pd.DataFrame({'stim_id':stim_id, 'region':region, 'first_cell_id':strong_pair[0], 'second_cell_id':strong_pair[1], 'corr_coef':corr_coef, 'p_value':p_value, 'bin_length':bin_length}, index=[0])
    correlations_by_bin_length_file = os.path.join(csv_dir, 'correlations_by_bin_length.csv')
    if not(os.path.isfile(correlations_by_bin_length_file)):
        corr_info_frame.to_csv(correlations_by_bin_length_file, index=False)
    else:
        corr_info_frame.to_csv(correlations_by_bin_length_file, header=False, mode='a', index=False)
    return pd.read_csv(correlations_by_bin_length_file)
