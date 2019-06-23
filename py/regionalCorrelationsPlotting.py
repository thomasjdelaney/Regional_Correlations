import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

regions = ['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1']
colours = cm.gist_rainbow(np.linspace(0, 1, 5)) # 5 regions
region_to_colour = dict(list(zip(regions, colours)))

def plotCorrMatrix(corr_matrix, cell_info, region_sorted_cell_ids):
    plt.matshow(corr_matrix)
    cell_range = np.arange(corr_matrix.shape[0])
    cell_regions = cell_info.loc[region_sorted_cell_ids]['region'].values
    plt.xticks(cell_range, cell_regions)
    plt.yticks(cell_range, cell_regions)
    plt.setp(plt.xticks()[1], rotation=-45, ha="right", rotation_mode="anchor")
    plt.colorbar()
    plt.tight_layout()

def plotRasterWithCellsAndTimes(region_sorted_cell_ids, spike_time_dict, start_time, stop_time, trials_info, cell_info):
    num_cells = region_sorted_cell_ids.size
    relevant_trials_info = trials_info[(start_time < trials_info[:,1]) & (trials_info[:,0] < stop_time),:]
    relevant_cell_info = cell_info.loc[region_sorted_cell_ids]
    for i, cell_id in enumerate(region_sorted_cell_ids):
        spike_times = spike_time_dict[cell_id][(start_time < spike_time_dict[cell_id]) & (spike_time_dict[cell_id] < stop_time)]
        if spike_times.size > 0:
            plt.vlines(x=spike_times, ymin=i+0.05, ymax=i+0.95, color=region_to_colour[relevant_cell_info.loc[cell_id]['region']], alpha=1.0)
    plt.ylim([0, num_cells])
    for trial_info in relevant_trials_info:
        if trial_info[2] != 17:
            trial_start = trial_info[0]; trial_stop = trial_info[1];
            plt.fill_between(x=[trial_start, trial_stop], y1=0, y2=num_cells, color='black', alpha=0.2)
    plt.xlabel('Time (s)', fontsize='large')
    plt.xlim([start_time, stop_time])
    y_ticks = np.array([np.flatnonzero(relevant_cell_info['region'] == r).mean() for r in relevant_cell_info['region'].unique()])
    tick_labels = np.array([r.replace('_', ' ').capitalize() for r in relevant_cell_info['region'].unique()])
    plt.yticks(y_ticks, tick_labels, fontsize='large')
    plt.tight_layout()

