import os, sys, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product, combinations
if float(sys.version[:3])<3.0:
    from pyentropy import DiscreteSystem
from scipy.stats import pearsonr

regions = ['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1']
bin_widths = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

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

def getRandomSelection(cell_info, trials_info, num_cells, group, probe, region, posterior_dir, frontal_dir, id_adjustor, is_weak=False, strong_threshold=20.0):
    # Get a random selection of cells. Filtering by group, probe, and region is possible.
    is_group = np.array([g in group for g in cell_info['group']])
    is_probe = np.array([p in probe for p in cell_info['probe']])
    is_region = np.array([r in region for r in cell_info['region']])
    all_cells = cell_info.index[is_region & (is_group & is_probe)]
    spike_time_dict = loadSpikeTimes(posterior_dir, frontal_dir, all_cells, id_adjustor)
    chosen_cells = getRespondingCells(all_cells, trials_info, spike_time_dict, cell_info, num_cells=num_cells, is_weak=is_weak, strong_threshold=strong_threshold)
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

def getStimTimesIds(stim_info, stim_id=0):
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

def excludeBumTrials(bin_array, trials_info):
    # WARN: If the number of 'bum trials' exceeds the number of 'good trials', this function will cause problems
    sizes = np.array([b.size for b in bin_array])
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    common_size = unique_sizes[counts.argmax()]
    bum_trial_inds = np.where(sizes != common_size)[0]
    num_bum_trials = bum_trial_inds.size
    print(dt.datetime.now().isoformat() + ' WARN: ' + 'Found ' + str(num_bum_trials) + ' bum trials...')
    new_bin_array = np.vstack(np.delete(bin_array, bum_trial_inds))
    new_trials_info = np.delete(trials_info, bum_trial_inds, axis=0)
    return new_bin_array, new_trials_info

def getTrialsFrame(trials_info, bin_width):
    bin_array = np.array([getBinTimes(stim_start, stim_stop, bin_width) for stim_start, stim_stop, stim_id in trials_info])
    if bin_array.dtype == 'O': # if some trials have a different number of bins, we need to exclude those trials
        bin_array, trials_info = excludeBumTrials(bin_array, trials_info)
    num_trials, num_bins = bin_array.shape
    num_bins -= 1
    trials_frame = pd.DataFrame(columns=['stim_start', 'stim_stop', 'bin_start', 'bin_stop', 'stim_id'])
    for i, bin_times in enumerate(bin_array):
        stim_start, stim_stop, stim_id = trials_info[i]
        trials_frame = trials_frame.append(pd.DataFrame({'stim_start':stim_start.repeat(num_bins), 'stim_stop':stim_stop.repeat(num_bins), 'bin_start':bin_times[0:num_bins], 'bin_stop':bin_times[1:num_bins+1], 'stim_id':stim_id.repeat(num_bins)}), ignore_index=True)
    return trials_frame, num_bins, num_trials

def getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, bin_width):
    # returns a frame with number of rows = number of bins X number of trials X number of cells X number of stimuli
    # each row contains the number of spikes in the associated bin.
    exp_frame = pd.DataFrame(columns=['stim_id', 'stim_start', 'stim_stop', 'cell_id', 'bin_start', 'bin_stop', 'num_spikes'])
    if all(cell_ids == 0):
        return exp_frame
    trials_frame, num_bins, num_trials = getTrialsFrame(trials_info, bin_width)
    bin_stops = trials_frame.bin_stop.values[[i*num_bins-1 for i in np.arange(1,num_trials+1)]]
    starts_and_stops = np.hstack([trials_frame.bin_start.values, bin_stops])
    starts_and_stops.sort()
    for cell_id in cell_ids:
        spike_counts = np.histogram(spike_time_dict[cell_id], starts_and_stops)[0]
        spike_counts = np.delete(spike_counts, [np.where(starts_and_stops == bin_stop)[0][0] for bin_stop in bin_stops[:-1]])
        trials_frame['num_spikes'] = spike_counts
        trials_frame['cell_id'] = np.repeat(cell_id, spike_counts.size)
        exp_frame = exp_frame.append(trials_frame, ignore_index=True)
    for col in ['stim_id', 'cell_id', 'num_spikes']:
        exp_frame[col] = exp_frame[col].astype(int)
    exp_frame = exp_frame.join(cell_info[['region', 'probe', 'depth']], on='cell_id')
    return exp_frame.sort_values(['probe', 'region', 'depth'])

def getRespondingCells(cell_ids, trials_info, spike_time_dict, cell_info, num_cells=0, is_weak=False, strong_threshold=20.0):
    big_frame = getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, 0.0)
    agg_frame = big_frame[['cell_id', 'num_spikes']].groupby('cell_id').agg('mean').sort_values('num_spikes', ascending=False)
    strongly_responding_cells = agg_frame.index[agg_frame['num_spikes'] >= strong_threshold].values
    weakly_responding_cells = agg_frame.index[(agg_frame['num_spikes'] <= strong_threshold)&(agg_frame['num_spikes'] > 0.0)].values
    responding_cells = weakly_responding_cells if is_weak else strongly_responding_cells
    num_responding_cells = responding_cells.size
    if num_cells > 0: # we only want a certain number of cells
        if num_responding_cells > num_cells:
            chosen_cells = np.random.choice(responding_cells, num_cells, replace=False)
        else:
            print(dt.datetime.now().isoformat() + ' WARN: ' + 'Only ' + str(num_responding_cells) + ' cells found.')
            chosen_cells = responding_cells
    else: # we want all cells
        chosen_cells = responding_cells
    return chosen_cells

def getRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, num_pairs, is_weak=False, strong_threshold=20.0):
    responding_cells = getRespondingCells(cell_ids, trials_info, spike_time_dict, cell_info, num_cells=0, is_weak=is_weak, strong_threshold=strong_threshold)
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

def getBestStimFromRegion(correlation_frame, region):
    region_frame = correlation_frame[correlation_frame.region == region]
    best_stim = region_frame['stim_id'].value_counts().index[0]
    if best_stim == 17:
        best_stim = region_frame['stim_id'].value_counts().index[1]
    return best_stim

def getMISymmUnc(discrete_system):
    info = np.max([0, discrete_system.I()])
    symm_unc = 2*info/(discrete_system.H['HX'] + discrete_system.H['HY'])
    return info, symm_unc

def calcInfo(discrete_system):
    discrete_system.calculate_entropies(method='plugin', calc=['HX', 'HXY', 'HY'])
    plugin = getMISymmUnc(discrete_system)
    discrete_system.calculate_entropies(method='qe', calc=['HX', 'HXY', 'HY'])
    qe = getMISymmUnc(discrete_system)
    return np.hstack([plugin, qe])

def getMutualInfoFromPair(pair, exp_frame):
    first_response = exp_frame[exp_frame.cell_id == pair[0]]['num_spikes']
    second_response = exp_frame[exp_frame.cell_id == pair[1]]['num_spikes']
    first_response_dims = [1, 1 + first_response.max()]
    second_response_dims = [1, 1 + second_response.max()]
    discrete_system = DiscreteSystem(first_response, first_response_dims, second_response, second_response_dims)
    return calcInfo(discrete_system)

def getCorrCoefFromPair(pair, exp_frame):
    first_response = exp_frame[exp_frame.cell_id == pair[0]]['num_spikes']
    second_response = exp_frame[exp_frame.cell_id == pair[1]]['num_spikes']
    return pearsonr(first_response, second_response)
