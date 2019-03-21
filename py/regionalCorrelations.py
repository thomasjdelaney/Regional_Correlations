import os
import numpy as np
import pandas as pd
from itertools import product

def loadCellInfo(csv_dir):
    # load the csv containing information about the cells. Creates a unique ID for each cell
    from_file = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'))
    max_posterior_cluster_id = from_file['cluster_id'][from_file['probe']=='posterior'].max()
    id_adjustor = max_posterior_cluster_id+1
    adj_cluster_id = from_file['cluster_id'].values
    adj_cluster_id[from_file['probe']=='frontal'] += id_adjustor
    from_file['adj_cluster_id'] = adj_cluster_id
    cell_info = from_file.set_index('adj_cluster_id')
    return cell_info, id_adjustor

def getRandomSelection(cell_info, num_cells, group, probe, region):
    # Get a random selection of cells. Filtering by group, probe, and region is possible.
    is_group = np.array([g in group for g in cell_info['group']])
    is_probe = np.array([p in probe for p in cell_info['probe']])
    is_region = np.array([r in region for r in cell_info['region']])
    chosen_cells = np.random.choice(cell_info.index[is_region & (is_group & is_probe)], size=num_cells, replace=False)
    chosen_cells.sort()
    return chosen_cells

def loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor):
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
        num_bins = bin_times.size-1
        spike_counts = np.histogram(spike_time_dict[cell_id], bins=bin_times)[0]
        exp_frame = exp_frame.append(pd.DataFrame({'stim_id':stim_id.repeat(num_bins), 'stim_start':stim_start.repeat(num_bins), 'stim_stop':stim_stop.repeat(num_bins), 'cell_id':cell_id.repeat(num_bins), 'bin_start':bin_times[0:num_bins], 'bin_stop':bin_times[1:num_bins+1], 'num_spikes':spike_counts}), ignore_index=True)
    for col in ['stim_id', 'cell_id', 'num_spikes']:
        exp_frame[col] = exp_frame[col].astype(int)
    exp_frame = exp_frame.join(cell_info[['region', 'probe', 'depth']], on='cell_id')
    return exp_frame.sort_values(['probe', 'region', 'depth'])
