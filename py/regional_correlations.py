"""
For calculating the pairwise correlations between many neurons, then clustering based on those correlations,
then comparing these clusters to the biological paritioning of the cells.
    exec(open(os.path.join(os.environ['HOME'], '.pystartup')).read())
"""
import os, argparse, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.io import loadmat
from itertools import product
from scipy.stats.stats import pearsonr

parser = argparse.ArgumentParser(description='Calculate pairwise correlations between given choice of neurons.')
parser.add_argument('-c', '--cell_choice', help='The method of choosing the cells.', default='random', choices=['random', 'specified'])
parser.add_argument('-n', '--number_of_cells', help='The number of cells to choose at random.', default=10, type=int)
parser.add_argument('-i', '--cell_ids', help='List of cell ids. (used when cell_choice is "specified")', nargs='*', type=int, default=[1689,  286,   84, 1791,  671, 1000, 1198, 1367,  762, 1760])
parser.add_argument('-g', '--group', help='The quality of sorting for randomly chosen_cells.', default=['good', 'mua', 'unsorted'], type=str, nargs='*')
parser.add_argument('-p', '--probe', help='Filter the randomly chosen cells by probe', default=['posterior', 'frontal'], type=str, nargs='*')
parser.add_argument('-r', '--region', help='Filter the randomly chosen cells by region', default=['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1'], type=str, nargs='*')
parser.add_argument('-b', '--bin_length', help='The length of time bins used, in seconds.', default=0.0, type=float)
parser.add_argument('-m', '--plot_correlation_matrix', help='Flag to plot the correlation matrix', default=False, action='store_true')
parser.add_argument('-a', '--correlation_figure_filename', help='Where to save the correlation matrix plot.', default='', type=str)
parser.add_argument('-e', '--save_correlation_with_bin_length', help='Flag to save a strong pairwise correlation', default=False, action='store_true')
parser.add_argument('-j', '--stim_id', help='A stim_id for use in the correlations vs bin length.', default=2, type=int)
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
args = parser.parse_args()

np.random.seed(args.numpy_seed) # setting seed
pd.set_option('max_rows',30) # setting display options for terminal display

# defining useful directories
proj_dir = os.path.join(os.environ['HOME'], 'Regional_Correlations')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
image_dir = os.path.join(proj_dir, 'images')

# loading stimulus info TODO: think about using the other stimuli
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))

def getRandomSelection(cell_info, num_cells, group, probe, region):
    # Get a random selection of cells. Filtering by group, probe, and region is possible.
    is_group = np.array([g in group for g in cell_info['group']])
    is_probe = np.array([p in probe for p in cell_info['probe']])
    is_region = np.array([r in region for r in cell_info['region']])
    chosen_cells = np.random.choice(cell_info.index[is_region & (is_group & is_probe)], size=num_cells, replace=False)
    chosen_cells.sort()
    return chosen_cells

def loadCellInfo():
    # load the csv containing information about the cells. Creates a unique ID for each cell
    from_file = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'))
    max_posterior_cluster_id = from_file['cluster_id'][from_file['probe']=='posterior'].max()
    id_adjustor = max_posterior_cluster_id+1
    adj_cluster_id = from_file['cluster_id'].values
    adj_cluster_id[from_file['probe']=='frontal'] += id_adjustor
    from_file['adj_cluster_id'] = adj_cluster_id
    cell_info = from_file.set_index('adj_cluster_id')
    return cell_info, id_adjustor

def loadSpikeTimes(cell_ids, id_adjustor):
    # load the spike times (s) of the given cells. Uses id_adjustor to convert the probe IDs to our unique IDs
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

def getStimTimesIds(stim_info, stim_id):
    # get stim start times, stim stop times, and stim IDs
    if stim_id == 0:
        trials_info = np.vstack([stim_info['stimStarts'][0], stim_info['stimStops'][0], stim_info['stimIDs'][0]]).T
    else:
        is_wanted_id = stim_info['stimIDs'][0] == stim_id
        starts = stim_info['stimStarts'][0][is_wanted_id]
        stops = stim_info['stimStops'][0][is_wanted_id]
        ids = stim_info['stimIDs'][0][is_wanted_id]
        trials_info = np.vstack([starts, stops, ids]).T
    return trials_info

def getBinTimes(stim_start, stim_stop, bin_width):
    # separate the period from stim_start to stim_stop into bins of width bin_width
    total_time = stim_stop - stim_start
    if (total_time == bin_width)|(bin_width == 0.0):
        bin_times = np.array([stim_start, stim_stop])
    elif (total_time/bin_width).is_integer():
        bin_times = np.hstack([np.arange(stim_start, stim_stop, bin_width), stim_stop])
    else:
        bin_times = np.arange(stim_start, stim_stop, bin_width)
    return bin_times

def getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, bin_width):
    # returns a frame with number of rows = number of bins X number of trials X number of cells X number of stimuli
    # each row contains the number of spikes in the associated bin.
    exp_frame = pd.DataFrame(columns=['stim_id', 'stim_start', 'stim_stop', 'cell_id', 'bin_start', 'bin_stop', 'num_spikes'])
    for cell_id, trial_info in product(cell_ids, trials_info):
        stim_start, stim_stop, stim_id = trial_info
        bin_times = getBinTimes(stim_start, stim_stop, bin_width)
        spike_counts = np.histogram(spike_time_dict[cell_id], bins=bin_times)[0]
        for i, bin_start in enumerate(bin_times[:-1]):
            exp_frame = exp_frame.append({'stim_id':stim_id, 'stim_start':stim_start, 'stim_stop':stim_stop, 'cell_id':cell_id, 'bin_start':bin_start, 'bin_stop':bin_times[i+1], 'num_spikes':spike_counts[i]}, ignore_index=True)
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

def getStronglyRespondingPair(exp_frame, cell_info, bin_width, stim_id, region):
    mean_responses = exp_frame[(exp_frame.stim_id == stim_id)&(exp_frame.region == region)][['cell_id', 'num_spikes']].groupby('cell_id').agg({'num_spikes':'mean'})
    mean_responses = mean_responses[mean_responses.num_spikes >= 10*bin_width].sort_values('num_spikes', ascending=False)
    if mean_responses.shape[0] < 2:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Less than 2 strongly responding neurons. Exiting.')
    return mean_responses.index.values[:2]

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

def plotCorrMatrixForStim(corr_matrix, cell_info, region_sorted_cell_ids):
    plt.matshow(corr_matrix)
    cell_range = np.arange(correlation_dict[2][0].shape[0])
    cell_regions = cell_info.loc[region_sorted_cell_ids]['region'].values
    plt.xticks(cell_range, cell_regions)
    plt.yticks(cell_range, cell_regions)
    plt.setp(plt.xticks()[1], rotation=-45, ha="right", rotation_mode="anchor")
    plt.colorbar()
    plt.tight_layout()

def showCellInfoTable(cell_info, region_sorted_cell_ids):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False); ax.axis('off'); ax.axis('tight'); # hide axes
    ax.table(cellText=cell_info.loc[region_sorted_cell_ids].values, colLabels=cell_info.columns, loc='center', fontsize='large')
    fig.tight_layout()

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
cell_info, id_adjustor = loadCellInfo()
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Selecting cells...')
cell_ids = getRandomSelection(cell_info, args.number_of_cells, args.group, args.probe, args.region)
spike_time_dict = loadSpikeTimes(cell_ids, id_adjustor)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading trial info...')
trials_info = getStimTimesIds(stim_info, args.stim_id)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Creating experiment frame...')
exp_frame = getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, args.bin_length)
if args.plot_correlation_matrix:# show example correlation matrix for all stimuli
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Measuring correlations...')
    correlation_dict = {}
    for stim_id in np.unique(stim_info['stimIDs'][0]).astype(int):
        correlation_dict[stim_id] = getCorrelationMatrixForStim(exp_frame, stim_id)
    correlation_dict[-1] = getCorrelationMatrixForStim(exp_frame, -1) # all stims at once
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting correlations...')
    plotCorrMatrixForStim(correlation_dict[-1][0], cell_info, exp_frame['cell_id'].unique())
    if args.correlation_figure_filename == '':
        plt.show(block=False)
        showCellInfoTable(cell_info, exp_frame['cell_id'].unique()) # show cell info in table
        plt.show(block=False)
    else:
        plt.savefig(os.path.join(image_dir, 'pairwise_correlation_matrices', args.correlation_figure_filename))
if args.save_correlation_with_bin_length:
    corr_info = saveStronglyRespondingCorrelation(exp_frame, cell_info, args.bin_length, args.stim_id, args.region[0])
