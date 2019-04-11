import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Calculate pairwise correlations between given choice of neurons.')
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
parser.add_argument('-g', '--group', help='The quality of sorting for randomly chosen_cells.', default='good', choices=['good', 'mua', 'unsorted'], type=str)
parser.add_argument('-a', '--is_strong', help='Flag for strongly or weakly responding cells', default=True, action='store_false')
parser.add_argument('-f', '--filename', help='Name of file for saving the csv.', type=str, default='all_regions_stims_pairs_widths.csv')
parser.add_argument('-t', '--threshold', help='Threshold spike count for trial classifying a cell as "strongly responding".', type=float, default=20.0)
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

def getFiringRatesForWidth(responding_cells, trials_info, spike_time_dict, cell_info, bin_width):
    print(dt.datetime.now().isoformat() + ' INFO: Calculating firing rates for bin width = ' + str(bin_width) + '...')
    responding_frame = rc.getExperimentFrame(responding_cells, trials_info, spike_time_dict, cell_info, bin_width)
    responding_frame.loc[:,'firing_rate'] = responding_frame.loc[:,'num_spikes']/bin_width
    agg_frame = responding_frame[['region', 'cell_id', 'stim_id', 'firing_rate']].groupby(['region', 'cell_id', 'stim_id']).agg(['mean','std','count'])
    agg_frame.reset_index(col_level=1, inplace=True)
    agg_frame.columns = agg_frame.columns.droplevel(level=0)
    agg_frame.columns = ['region', 'cell_id', 'stim_id', 'firing_rate_mean', 'firing_rate_std', 'num_samples']
    agg_frame.loc[:,'bin_width'] = bin_width
    return agg_frame

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
    stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
    cell_ids = cell_info[cell_info.group==args.group].index.values
    trials_info = rc.getStimTimesIds(stim_info)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting responding cells...')
    spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
    responding_cells = rc.getRespondingCells(cell_ids, trials_info, spike_time_dict, cell_info, num_cells=0, is_strong=args.is_strong, strong_threshold=20.0)
    num_responding_cells = responding_cells.size
    spike_time_dict = {k: spike_time_dict[k] for k in responding_cells}
    all_rates = pd.concat([getFiringRatesForWidth(responding_cells, trials_info, spike_time_dict, cell_info, bin_width) for bin_width in rc.bin_widths], ignore_index=True)
    all_rates.to_csv(os.path.join(csv_dir, args.filename))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
