"""
For calculating the signal correlations between many neurons using different bin widths.
"""
import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from scipy.io import loadmat
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='For signal correlation vs time bin plots. ')
parser.add_argument('-g', '--group', help='The quality of sorting for randomly chosen_cells.', default='good', choices=['good', 'mua', 'unsorted'], type=str, nargs='*')
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed) # setting seed
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

def getBinWidthSignalCorrFrame(responding_pairs, trials_info, spike_time_dict, cell_info, bin_width, region):
    num_pairs = responding_pairs.shape[0]
    responding_cells = np.unique(responding_pairs)
    exp_frame = rc.getExperimentFrame(responding_cells, trials_info, spike_time_dict, cell_info, bin_width)
    agg_frame = exp_frame[['cell_id', 'stim_id', 'num_spikes']].groupby(['cell_id', 'stim_id']).agg('mean')
    correlations = np.zeros(num_pairs)
    p_values = np.zeros(num_pairs)
    for j, pair in enumerate(responding_pairs):
        correlations[j], p_values[j] = pearsonr(agg_frame.loc[pair[0]], agg_frame.loc[pair[1]])
    return pd.DataFrame({'region':np.repeat(region, num_pairs), 'first_cell_id':responding_pairs[:,0], 'second_cell_id':responding_pairs[:,1], 'signal_corr_coef':correlations, 'p_values':p_values, 'bin_width':np.repeat(bin_width, num_pairs)})

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
    stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
    trials_info = rc.getStimTimesIds(stim_info)
    signal_corr_frame = pd.DataFrame(columns=['region', 'first_cell_id', 'second_cell_id', 'signal_corr_coef', 'p_values', 'bin_width'])
    for region in rc.regions:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Making signal correlation frame for '+ region + '...')
        cell_ids = cell_info[(cell_info.region==region)&(cell_info.group==args.group)].index.values
        spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
        responding_pairs = rc.getRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, 30, is_strong=True)
        for bin_width in rc.bin_widths:
            signal_corr_frame = signal_corr_frame.append(getBinWidthSignalCorrFrame(responding_pairs, trials_info, spike_time_dict, cell_info, bin_width, region))
    csv_filename = os.path.join(csv_dir, 'signal_correlations_strong.csv')
    signal_corr_frame.to_csv(csv_filename, index=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + csv_filename + ' saved.')

if not(args.debug):
    main()
