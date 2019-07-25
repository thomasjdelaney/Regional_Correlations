"""
Script for detecting communities in the networks created by measuring correlations and mutual information.
"""
import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import glob
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.io import loadmat
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser(description='For creating histograms of correlation coefficients.')
parser.add_argument('-n', '--number_of_cells', help='The number of cells to choose at random.', default=1000, type=int)
parser.add_argument('-b', '--bin_width', help='The bin width to use for correlations.', type=float, default=1.0)
parser.add_argument('-g', '--group', help='The quality of sorting for randomly chosen_cells.', default=['good'], type=str, nargs='*')
parser.add_argument('-p', '--probe', help='Filter the randomly chosen cells by probe', default=['posterior', 'frontal'], type=str, nargs='*')
parser.add_argument('-r', '--region', help='Filter the randomly chosen cells by region', default=['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1'], type=str, nargs='*')
parser.add_argument('-j', '--stim_id', help='A stim_id for use in the correlations vs bin length.', default=2, type=int)
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
parser.add_argument('-x', '--prefix', help='A prefix for the image file names.', type=str, default='')
parser.add_argument('-a', '--percentile', help='Percentile to use when sparsifying measure matrix', type=float, default=50.0)
parser.add_argument('-f', '--numpy_file_prefix', help='If used, indicates a set of files containing saved data.', type=str, default='')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

np.random.seed(args.numpy_seed) # setting seed
np.set_printoptions(linewidth=200) # make it easier to see numpy arrays
pd.set_option('max_rows',30) # setting display options for terminal display
pd.options.mode.chained_assignment = None  # default='warn'

# defining useful directories
proj_dir = os.path.join(os.environ['PROJ'], 'Regional_Correlations')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
npy_dir = os.path.join(proj_dir, 'npy')
image_dir = os.path.join(proj_dir, 'images')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
mat_dir = os.path.join(proj_dir, 'mat')

# loading useful functions
sys.path.append(py_dir)
sys.path.append(os.environ['PROJ'])
import regionalCorrelations as rc
import regionalCorrelationsPlotting as rcp
import Network_Noise_Rejection_Python as nnr

def getPairwiseMeasurementFrame(pairs, exp_frame, cell_info, stim_id, bin_width):
    mutual_infos = np.zeros((pairs.shape[0], 4))
    correlation_coefficients = np.zeros(pairs.shape[0])
    p_values = np.zeros(pairs.shape[0])
    for i, pair in enumerate(pairs):
        # print(dt.datetime.now().isoformat() + ' INFO: ' + ' processing pair number ' + str(i) + '...')
        correlation_coefficients[i], p_values[i] = rc.getCorrCoefFromPair(pair, exp_frame)
        mutual_infos[i] = rc.getMutualInfoFromPair(pair, exp_frame)
    return pd.DataFrame({'stim_id':np.repeat(stim_id, pairs.shape[0]), 'first_region':cell_info.loc[pairs[:,0],'region'].values, 'second_region':cell_info.loc[pairs[:,1],'region'].values, 'first_cell_id':pairs[:,0], 'second_cell_id':pairs[:,1], 'mutual_info_plugin':mutual_infos[:,0], 'symm_unc_plugin':mutual_infos[:,1], 'mutual_info_qe':mutual_infos[:,2], 'symm_unc_qe':mutual_infos[:,3], 'corr_coef':correlation_coefficients, 'p_value':p_values, 'bin_width':np.repeat(bin_width, pairs.shape[0])})

def getPairwiseMeasurementMatrices(pairs, region_sorted_cell_ids, pairwise_measurements):
    num_cells = region_sorted_cell_ids.size
    corr_matrix = np.zeros([num_cells, num_cells])
    symm_unc_matrix = np.zeros([num_cells, num_cells])
    info_matrix = np.zeros([num_cells, num_cells])
    for pair in pairs:
        first_index = region_sorted_cell_ids.tolist().index(pair[0])
        second_index = region_sorted_cell_ids.tolist().index(pair[1])
        pair_record = pairwise_measurements[(pairwise_measurements.first_cell_id == pair[0]) & (pairwise_measurements.second_cell_id == pair[1])].iloc[0]
        corr_matrix[first_index, second_index] = corr_matrix[second_index, first_index] = pair_record.corr_coef
        symm_unc_matrix[first_index, second_index] = symm_unc_matrix[second_index, first_index] = pair_record.symm_unc_qe
        info_matrix[first_index, second_index] = info_matrix[second_index, first_index] = pair_record.mutual_info_qe
    return corr_matrix, symm_unc_matrix, info_matrix

def sparsifyMeasureMatrix(measure_matrix, percentile):
    threshold = np.percentile(measure_matrix[measure_matrix.nonzero()], percentile)
    measure_matrix[measure_matrix < threshold] = 0
    return measure_matrix

def runNNRSaveFigsData(measure_matrix, measure_type): # measure type, 'info' or 'corr'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sparsifying info matrix...')
    measure_matrix = sparsifyMeasureMatrix(measure_matrix, args.percentile)

    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Checking the data is symmetric...')
    measure_matrix = nnr.checkDirected(measure_matrix)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting the biggest component...')
    measure_matrix, keep_indices, comp_assign, comp_size = nnr.getBiggestComponent(measure_matrix)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling from null model...')
    samples_eig_vals, optional_returns = nnr.getPoissonWeightedConfModel(measure_matrix, 100, return_eig_vecs=True, is_sparse=True)
    samples_eig_vecs = optional_returns['eig_vecs']
    expected_wcm = optional_returns['expected_wcm']

    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing network modularity matrix...')
    network_modularity_matrix = measure_matrix - expected_wcm

    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting low dimensional space...')
    below_eig_space, below_lower_bound_inds, [mean_mins_eig, min_confidence_ints], exceeding_eig_space, exceeding_upper_bound_inds, [mean_maxs_eig, max_confidence_ints] = nnr.getLowDimSpace(network_modularity_matrix, samples_eig_vals, 0, int_type='CI')
    exceeding_space_dims = exceeding_eig_space.shape[1]
    below_space_dims = below_eig_space.shape[1]

    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Splitting network into noise and signal...')
    reject_dict = nnr.nodeRejection(network_modularity_matrix, samples_eig_vals, 0, samples_eig_vecs, weight_type='linear', norm='L2', int_type='CI', bounds='upper')
    signal_weighted_adjacency_matrix = measure_matrix[reject_dict['signal_inds']][:, reject_dict['signal_inds']]

    if reject_dict['signal_inds'].size == 0:
        noise_final_cell_info = cell_info.loc[region_sorted_cell_ids[reject_dict['noise_inds']]]
        noise_final_cell_info.to_pickle(os.path.join(npy_dir, measure_type, 'noise_final_cell_info_' + args.numpy_file_prefix + '_' + str(int(args.percentile)) + '.pkl'))
        return 0

    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing final signal network without leaves...')
    biggest_signal_comp, biggest_signal_inds, biggest_signal_assing, biggest_signal_size = nnr.getBiggestComponent(signal_weighted_adjacency_matrix)
    signal_comp_inds = reject_dict['signal_inds'][biggest_signal_inds]
    degree_distn = (biggest_signal_comp > 0).sum(axis=0)
    leaf_inds = np.flatnonzero(degree_distn == 1)
    keep_inds = np.flatnonzero(degree_distn > 1)
    signal_final_inds = signal_comp_inds[keep_inds]
    signal_leaf_inds = signal_comp_inds[leaf_inds]
    final_weighted_adjacency_matrix = biggest_signal_comp[keep_inds][:, keep_inds]
    signal_final_cell_ids = region_sorted_cell_ids[signal_final_inds]
    signal_final_cell_info = cell_info.loc[signal_final_cell_ids]
    noise_final_cell_info = cell_info.loc[region_sorted_cell_ids[reject_dict['noise_inds']]]

    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Detecting communities...')
    signal_expected_wcm = expected_wcm[signal_final_inds][:, signal_final_inds]
    max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations = nnr.consensusCommunityDetect(final_weighted_adjacency_matrix, signal_expected_wcm, exceeding_space_dims+1, exceeding_space_dims+1)
    signal_final_cell_info['consensus_cluster'] = consensus_clustering
    nnr.plotClusterMap(final_weighted_adjacency_matrix, consensus_clustering, is_sort=True) # node_labels=signal_final_cell_info['region'].values
    plt.savefig(os.path.join(image_dir, 'community_detection', measure_type, 'cons_cluster_map_' + args.numpy_file_prefix + '_' + str(int(args.percentile)) + '.png'))
    plt.close()
    nnr.plotModEigValsVsNullEigHist(network_modularity_matrix, samples_eig_vals)
    plt.savefig(os.path.join(image_dir, 'community_detection', measure_type, 'eig_hist_' + args.numpy_file_prefix + '_' + str(int(args.percentile)) + '.png'))
    plt.close()
    nnr.plotModEigValsVsNullEig(network_modularity_matrix, mean_mins_eig, mean_maxs_eig)
    plt.savefig(os.path.join(image_dir, 'community_detection', measure_type, 'eig_plot_' + args.numpy_file_prefix + '_' + str(int(args.percentile)) + '.png'))
    plt.close()
    signal_final_cell_info.to_pickle(os.path.join(npy_dir, measure_type, 'signal_final_cell_info_' + args.numpy_file_prefix + '_' + str(int(args.percentile)) + '.pkl'))
    noise_final_cell_info.to_pickle(os.path.join(npy_dir, measure_type, 'noise_final_cell_info_' + args.numpy_file_prefix + '_' + str(int(args.percentile)) + '.pkl'))
    print(dt.datetime.now().isoformat() + ' INFO: ' + measure_type.capitalize() + ' done.')

def plotRegionalClusterMap(signal_final_cell_info):
    colours = rcp.region_to_colour.keys()
    num_nodes = signal_final_cell_info.shape[0]
    sorted_final_cell_info = signal_final_cell_info.sort_values(['consensus_cluster', 'region', 'depth'], ascending=[False, True, True])
    regional_cluster_matrix = np.zeros([num_nodes, num_nodes])
    for i,j in combinations(np.arange(num_nodes), 2):
        i_region = sorted_final_cell_info.iloc[i]['region']
        j_region = sorted_final_cell_info.iloc[j]['region']
        i_cluster = sorted_final_cell_info.iloc[i]['consensus_cluster']
        j_cluster = sorted_final_cell_info.iloc[j]['consensus_cluster']
        if (i_region == j_region) & (i_cluster == j_cluster):
            regional_cluster_matrix[i,j] = colours.index(i_region)+1
    regional_cluster_matrix = regional_cluster_matrix + regional_cluster_matrix.T
    sorted_clustering = sorted_final_cell_info['consensus_cluster'].values
    cluster_changes = np.hstack([-1, np.flatnonzero(np.diff(sorted_clustering) != 0), sorted_clustering.size-1]) + 0.5
    cmap = colors.ListedColormap(np.vstack([colors.to_rgba('paleturquoise'), rcp.region_to_colour.values()]))
    bounds = [0,1,2,3,4,5,6]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.figure()
    ax = plt.gca()
    im = ax.matshow(regional_cluster_matrix, cmap=cmap, norm=norm)
    for i in range(cluster_changes.size-2):
        ax.plot([cluster_changes[i+1], cluster_changes[i+1]], [cluster_changes[i], cluster_changes[i+2]], color='white', linewidth=2.0)
        ax.plot([cluster_changes[i], cluster_changes[i+2]], [cluster_changes[i+1], cluster_changes[i+1]], color='white', linewidth=2.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=np.array(bounds)[1:] - 0.5)
    cb.set_ticklabels([c.replace('_', ' ').capitalize() for c in [''] + colours] )
    plt.tight_layout()

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading trials info...')
trials_info = rc.getStimTimesIds(stim_info, args.stim_id)
if args.numpy_file_prefix != '':
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading data from disc...')
    region_sorted_cell_ids = np.load(os.path.join(npy_dir, args.numpy_file_prefix + '_responding_sorted_cell_ids.npy'), allow_pickle=True)
    pairs = np.load(os.path.join(npy_dir, args.numpy_file_prefix + '_responding_pairs.npy'), allow_pickle=True)
    pairwise_measurements = pd.read_pickle(os.path.join(npy_dir, args.numpy_file_prefix + '_pairwise_measurements.pkl'))
    info_matrix = np.load(os.path.join(npy_dir, args.numpy_file_prefix + '_info_matrix.npy'), allow_pickle=True)
    corr_matrix = np.load(os.path.join(npy_dir, args.numpy_file_prefix + '_corr_matrix.npy'), allow_pickle=True)
    symm_unc_matrix = np.load(os.path.join(npy_dir, args.numpy_file_prefix + '_symm_unc_matrix.npy'), allow_pickle=True)
else:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Selecting repsonding cells...')
    cell_ids = rc.getRandomSelection(cell_info, trials_info, args.number_of_cells, args.group, args.probe, args.region, posterior_dir, frontal_dir, id_adjustor, is_weak=False, strong_threshold=0.01)
    spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Creating experiment frame...')
    exp_frame = rc.getExperimentFrame(cell_ids, trials_info, spike_time_dict, cell_info, args.bin_width)
    region_sorted_cell_ids = exp_frame['cell_id'].unique()
    pairs = np.array(list(combinations(region_sorted_cell_ids, 2)))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Calculating correlation and information...')
    pairwise_measurements = getPairwiseMeasurementFrame(pairs, exp_frame, cell_info, args.stim_id, args.bin_width)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Creating symmetric matrices...')
    corr_matrix, symm_unc_matrix, info_matrix = getPairwiseMeasurementMatrices(pairs, region_sorted_cell_ids, pairwise_measurements)

# runNNRSaveFigsData(info_matrix, 'info')
# runNNRSaveFigsData(np.abs(corr_matrix), 'corr')
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
