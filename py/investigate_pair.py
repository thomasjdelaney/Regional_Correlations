"""
For investigating the relationship between a given pair of neurons.
"""
import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='For investigating the relationship between a given pair of neurons.')
parser.add_argument('-p', '--pair', help='The pair of ids to be investigated.', default=[3542, 3656], type=int, nargs=2)
parser.add_argument('-f', '--paired_file', help='The file in which to find the correlations and mutual info.', type=str, default='all_pairs.csv')
parser.add_argument('-g', '--single_file', help='The file in which to find the firing rates.', type=str, default='all_firing_rates.csv')
parser.add_argument('-b', '--bin_width', help='The bin width to use for correlations.', type=float, default=1.0)
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

# defining useful directories
proj_dir = os.path.join(os.environ['PROJ'], 'Regional_Correlations')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
image_dir = os.path.join(proj_dir, 'images')

# loading useful functions
sys.path.append(py_dir)
import regionalCorrelations as rc
import regionalCorrelationsPlotting as rcp

pd.set_option('max_rows', 30)

def getDataFrames(single_file, paired_file, bin_width):
    firing_frame = pd.read_csv(os.path.join(csv_dir, args.single_file), usecols=lambda x: x != 'Unnamed: 0')
    pairwise_frame = pd.read_csv(os.path.join(csv_dir, args.paired_file), usecols=lambda x: x != 'Unnamed: 0')
    firing_frame = firing_frame[firing_frame.bin_width == bin_width]
    pairwise_frame = pairwise_frame[pairwise_frame.bin_width == bin_width]
    return firing_frame, pairwise_frame

pair = np.array(args.pair)
cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
firing_frame, pairwise_frame = getDataFrames(args.single_file, args.paired_file, args.bin_width)
region = firing_frame.loc[firing_frame.cell_id == pair[0]].region.unique()[0]
best_stim = rc.getBestStimFromRegion(pairwise_frame, region)
pairwise_region_frame = pairwise_frame[(pairwise_frame.region == region) & (pairwise_frame.stim_id == best_stim)]
stim_firing_frame = firing_frame.loc[firing_frame.stim_id == best_stim,:]
trials_info = rc.getStimTimesIds(stim_info, best_stim)
spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, pair, id_adjustor)
exp_frame = rc.getExperimentFrame(pair, trials_info, spike_time_dict, cell_info, 1.0)

corr = pairwise_region_frame.loc[(pairwise_region_frame.first_cell_id == pair[0]) & (pairwise_region_frame.second_cell_id == pair[1]), 'corr_coef'].iloc[0]
info = pairwise_region_frame.loc[(pairwise_region_frame.first_cell_id == pair[0]) & (pairwise_region_frame.second_cell_id == pair[1]), 'mutual_info_qe'].iloc[0]
first_firing_rate = stim_firing_frame.loc[firing_frame.cell_id == pair[0], 'firing_rate_mean'].iloc[0]
second_firing_rate = stim_firing_frame.loc[firing_frame.cell_id == pair[1], 'firing_rate_mean'].iloc[0]
geom_mean_firing_rate = np.sqrt(first_firing_rate * second_firing_rate)
first_responses = exp_frame.loc[exp_frame.cell_id == pair[0], 'num_spikes']
second_responses = exp_frame.loc[exp_frame.cell_id == pair[1], 'num_spikes']

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(first_responses, second_responses)
plt.xlabel('Spike counts, cell ' + str(pair[0]), fontsize='large')
plt.xlim([0, first_responses.max()+1])
plt.ylabel('Spike counts, cell ' + str(pair[1]), fontsize='large')
plt.ylim([0, second_responses.max()+1])
plt.title('Corr = ' + str(round(corr,2)) + ', Info = '+ str(round(info,2)))

plt.subplot(1,2,2)

plt.plot(np.arange(1,21), first_responses, color='blue', label='cell ' + str(pair[0]))
plt.plot(np.arange(1,21), second_responses, color='orange', label='cell ' + str(pair[1]))
plt.xlabel('Trial number', fontsize='large')
plt.xlim([0,21])
plt.xticks(np.arange(0,22), np.arange(0,22))
plt.ylabel('Spike Count', fontsize='large')
plt.title('Geometric Mean Firing Rate = ' + str(round(geom_mean_firing_rate,2)))
plt.legend(fontsize='large')

plt.suptitle('Region = ' + str(region), fontsize='large')

file_name = 'spike_count_investigation_' + str(pair[0]) + '_' + str(pair[1]) + '_' + region + '.png'
plt.savefig(os.path.join(image_dir, 'spike_count_investigation', file_name))
