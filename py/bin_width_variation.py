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
from itertools import combinations, product
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='Calculate pairwise correlations between given choice of neurons.')
parser.add_argument('-n', '--wanted_num_pairs', help='The number of strongly responding pairs to use.', default=30, type=int)
parser.add_argument('-g', '--group', help='The quality of sorting for randomly chosen_cells.', default='good', choices=['good', 'mua', 'unsorted'], type=str, nargs='*')
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
parser.add_argument('-a', '--is_strong', help='Flag for strongly or weakly responding cells', default=True, action='store_false')
parser.add_argument('-f', '--filename', help='Name of file for saving the csv.', type=str, default='all_regions_stims_pairs_widths.csv')
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

def getRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, num_pairs, is_strong, strong_threshold=20.0):
    big_frame = rc.getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, 0.0)
    agg_frame = big_frame[['cell_id', 'num_spikes']].groupby('cell_id').agg('mean').sort_values('num_spikes', ascending=False)
    strongly_responding_cells = agg_frame.index[agg_frame['num_spikes'] >= strong_threshold].values
    weakly_responding_cells = agg_frame.index[(agg_frame['num_spikes'] <= strong_threshold)&(agg_frame['num_spikes'] > 0.0)].values
    responding_cells = strongly_responding_cells if is_strong else weakly_responding_cells
    if responding_cells.size < 2:
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Less than 2 responding cells.')
        return np.array([0,0])
    all_pairs = np.array(list(combinations(responding_cells, 2)))
    num_all_pairs = all_pairs.shape[0]
    if num_all_pairs >= num_pairs:
        randomly_chosen_pairs = all_pairs[np.random.choice(np.arange(0,num_all_pairs), num_pairs, replace=False),:]
    else:
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Only ' + str(num_all_pairs) + ' pairs found.')
        randomly_chosen_pairs = all_pairs
    return randomly_chosen_pairs

def getCorrCoefFromPair(pair, exp_frame):
    first_response = exp_frame[exp_frame.cell_id == pair[0]]['num_spikes']
    second_response = exp_frame[exp_frame.cell_id == pair[1]]['num_spikes']
    return pearsonr(first_response, second_response)

def getCorrFrameForWidth(bin_width, pairs, trials_info, spike_time_dict, cell_info, stim_id, region):
    print(dt.datetime.now().isoformat() + ' INFO: Calculating correlations for bin width = ' + str(bin_width) + '...')
    cells = np.unique(pairs)
    num_pairs = pairs.shape[0]
    width_exp_frame = rc.getExperimentFrame(cells, trials_info, spike_time_dict, cell_info, bin_width)
    correlation_coefficients = np.zeros(num_pairs)
    p_values = np.zeros(num_pairs)
    for i, pair in enumerate(pairs):
        correlation_coefficients[i], p_values[i] = getCorrCoefFromPair(pair, width_exp_frame)
    return pd.DataFrame({'stim_id':np.repeat(stim_id, num_pairs), 'region':np.repeat(region, num_pairs), 'first_cell_id':pairs[:,0], 'second_cell_id':pairs[:,1], 'corr_coef':correlation_coefficients, 'p_value':p_values, 'bin_width':np.repeat(bin_width, num_pairs)})

def getAllWidthFrameForRegionStim(cell_info, stim_info, id_adjustor, region, stim_id, group, wanted_num_pairs, is_strong, bin_widths):
    print(dt.datetime.now().isoformat() + ' INFO: Getting correlations for all widths for region = ' + region + ', stim ID = ' + str(stim_id) + '...')
    cell_ids = cell_info[(cell_info.region==region)&(cell_info.group==group)].index.values
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading trial info...')
    trials_info = rc.getStimTimesIds(stim_info, stim_id)
    spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting pairs of responding cells...')
    responding_pairs = getRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, wanted_num_pairs, is_strong)
    num_pairs = responding_pairs.shape[0]
    responding_cells = np.unique(responding_pairs)
    spike_time_dict = {k: spike_time_dict[k] for k in responding_cells}
    return pd.concat([getCorrFrameForWidth(bin_width, responding_pairs, trials_info, spike_time_dict, cell_info, stim_id, region) for bin_width in bin_widths], ignore_index=True)

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
    stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
    stim_ids = np.unique(stim_info['stimIDs'][0])
    bin_widths = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    regions = ['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1']
    all_regions_stims_pairs_widths = pd.concat([getAllWidthFrameForRegionStim(cell_info, stim_info, id_adjustor, region, stim_id, args.group, args.wanted_num_pairs, args.is_strong, bin_widths) for region,stim_id in product(regions, stim_ids)], ignore_index=True)
    all_regions_stims_pairs_widths.to_csv(os.path.join(csv_dir, args.filename))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
