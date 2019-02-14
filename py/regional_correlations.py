"""
For calculating the pairwise correlations between many neurons, then clustering based on those correlations,
then comparing these clusters to the biological paritioning of the cells.
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from itertools import product
from scipy.stats.stats import pearsonr

parser = argparse.ArgumentParser(description='Calculate pairwise correlations between given choice of neurons.')
parser.add_argument('-c', '--cell_choice', help='The method of choosing the cells.', default='random', choices=['random', 'specified'])
parser.add_argument('-n', '--number_of_cells', help='The number of cells to choose at random.', default=10, type=int)
parser.add_argument('-i', '--cell_ids', help='List of cell ids. (used when cell_choice is "specified")', nargs='*', type=int, default=[1689,  286,   84, 1791,  671, 1000, 1198, 1367,  762, 1760])
parser.add_argument('-g', '--cell_group', help='The quality of sorting for randomly chosen_cells.', default=['good', 'mua', 'unsorted'], type=str, nargs='*')
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
args = parser.parse_args()

np.random.seed(args.numpy_seed)
pd.set_option('max_rows',30)

proj_dir = os.path.join(os.environ['SPACE'], 'Regional_Correlations')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')

cell_subset = np.array([])
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))

def getRandomSelection(cell_info, num_cells, cell_group):
    chosen_cells = np.random.choice(cell_info.index[[group in cell_group for group in cell_info['group']]], size=num_cells)
    chosen_cells.sort()
    return chosen_cells

def loadCellInfo():
    from_file = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'))
    max_posterior_cluster_id = from_file['cluster_id'][from_file['probe']=='posterior'].max()
    id_adjustor = max_posterior_cluster_id+1
    adj_cluster_id = from_file['cluster_id'].values
    adj_cluster_id[from_file['probe']=='frontal'] += id_adjustor
    from_file['adj_cluster_id'] = adj_cluster_id
    cell_info = from_file.set_index('adj_cluster_id')
    return cell_info, id_adjustor

def loadSpikeTimes(cell_ids, id_adjustor):
    frames_per_second = 30000.0
    time_correction = np.load(os.path.join(frontal_dir, 'time_correction.npy'))
    post_spike_times = np.load(os.path.join(posterior_dir, 'spike_times.npy')).flatten()/frames_per_second
    front_spike_times = np.load(os.path.join(frontal_dir, 'spike_times.npy')).flatten()/frames_per_second-time_correction[1]
    post_spike_clusters = np.load(os.path.join(posterior_dir, 'spike_clusters.npy')).flatten()
    front_spike_clusters = np.load(os.path.join(frontal_dir, 'spike_clusters.npy')).flatten()
    front_spike_clusters += id_adjustor
    spike_times = {}
    for cell_id in cell_ids:
        cell_post_spike_times = post_spike_times[post_spike_clusters == cell_id]
        cell_front_spike_times = front_spike_times[front_spike_clusters == cell_id]
        spike_times[cell_id] = np.concatenate([cell_post_spike_times, cell_front_spike_times])
    return spike_times

def getStimTimesIds(stim_info):
    return np.vstack([stim_info['stimStarts'][0], stim_info['stimStops'][0], stim_info['stimIDs'][0]]).T

def getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info):
    exp_frame = pd.DataFrame(columns=['stim_id', 'stim_start', 'stim_stop', 'cell_id', 'num_spikes'])
    for cell_id, trial_info in product(cell_ids, trials_info):
        stim_start, stim_stop, stim_id = trial_info
        num_spikes = ((spike_time_dict[cell_id] > stim_start) & (spike_time_dict[cell_id] < stim_stop)).sum()
        exp_frame = exp_frame.append({'stim_id':stim_id, 'stim_start':stim_start, 'stim_stop':stim_stop, 'cell_id':cell_id, 'num_spikes':num_spikes}, ignore_index=True)
    for col in ['stim_id', 'cell_id', 'num_spikes']:
        exp_frame[col] = exp_frame[col].astype(int)
    exp_frame = exp_frame.join(cell_info[['region', 'probe', 'depth']], on='cell_id')
    return exp_frame.sort_values(['probe', 'region', 'depth'])

def getCorrelationMatrixForStim(exp_frame, stim_id):
    region_sorted_cell_ids = exp_frame['cell_id'].unique()
    num_cells = region_sorted_cell_ids.size
    if stim_id == -1:
        stim_frame = exp_frame[['cell_id', 'num_spikes']]
    else:
        stim_frame = exp_frame[exp_frame['stim_id']==stim_id][['cell_id', 'num_spikes']]
    spike_count_dict = {}
    for cid in region_sorted_cell_ids:
        spike_count_dict[cid] = stim_frame[stim_frame['cell_id']==cid]['num_spikes'].values
    corr_matrix = np.zeros(num_cells*num_cells); p_value_matrix = np.zeros(num_cells*num_cells)
    for i, cells in enumerate(product(region_sorted_cell_ids, region_sorted_cell_ids)):
        corr_matrix[i], p_value_matrix[i] = pearsonr(spike_count_dict[cells[0]], spike_count_dict[cells[1]])
    corr_matrix = corr_matrix.reshape(num_cells, num_cells); p_value_matrix = p_value_matrix.reshape(num_cells, num_cells)
    np.fill_diagonal(corr_matrix, 0.0); np.fill_diagonal(p_value_matrix, 0.0);
    return corr_matrix, p_value_matrix

def plotCorrMatrixForStim(corr_matrix):
    plt.matshow(corr_matrix)
    plt.colorbar()
    plt.tight_layout()

def showCellInfoTable(cell_info, region_sorted_cell_ids):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False); ax.axis('off'); ax.axis('tight'); # hide axes
    ax.table(cellText=cell_info.loc[region_sorted_cell_ids].values, colLabels=cell_info.columns, loc='center')
    fig.tight_layout()

cell_info, id_adjustor = loadCellInfo()
cell_ids = getRandomSelection(cell_info, args.number_of_cells, args.cell_group)
spike_time_dict = loadSpikeTimes(cell_ids, id_adjustor)
trials_info = getStimTimesIds(stim_info)
exp_frame = getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info)
correlation_dict = {}
for stim_id in np.unique(trials_info[:,2]).astype(int):
    correlation_dict[stim_id] = getCorrelationMatrixForStim(exp_frame, stim_id)
correlation_dict[-1] = getCorrelationMatrixForStim(exp_frame, -1) # all stims at once
# show example correlation matrix for all stimuli
plotCorrMatrixForStim(correlation_dict[-1][0])
plt.show(block=False)
# show cell info in table
showCellInfoTable(cell_info, exp_frame['cell_id'].unique())
plt.show(block=False)
