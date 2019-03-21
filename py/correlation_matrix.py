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

# loading stimulus info TODO: think about using the other stimuli
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))

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

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Selecting cells...')
    cell_ids = rc.getRandomSelection(cell_info, args.number_of_cells, args.group, args.probe, args.region)
    spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading trial info...')
    trials_info = rc.getStimTimesIds(stim_info, args.stim_id)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Creating experiment frame...')
    exp_frame = rc.getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, args.bin_length)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Measuring correlations...')
    correlation_dict = {}
    for stim_id in np.unique(stim_info['stimIDs'][0]).astype(int):
        correlation_dict[stim_id] = getCorrelationMatrixForStim(exp_frame, stim_id)
    correlation_dict[-1] = getCorrelationMatrixForStim(exp_frame, -1) # all stims at once
    if args.plot_correlation_matrix:# show example correlation matrix for all stimuli
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting correlations...')
        plotCorrMatrixForStim(correlation_dict[-1][0], cell_info, exp_frame['cell_id'].unique())
        if args.correlation_figure_filename == '':
            plt.show(block=False)
            showCellInfoTable(cell_info, exp_frame['cell_id'].unique()) # show cell info in table
            plt.show(block=False)
        else:
            plt.savefig(os.path.join(image_dir, 'pairwise_correlation_matrices', args.correlation_figure_filename))

if not(args.debug):
    main()

if args.save_correlation_with_bin_length:
    corr_info = saveStronglyRespondingCorrelation(exp_frame, cell_info, args.bin_length, args.stim_id, args.region[0])
