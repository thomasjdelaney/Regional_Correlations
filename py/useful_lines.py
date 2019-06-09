import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import bct
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.io import loadmat
from itertools import combinations
from scipy import stats
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='For creating histograms of correlation coefficients.')
parser.add_argument('-n', '--number_of_cells', help='The number of cells to choose at random.', default=10, type=int)
parser.add_argument('-b', '--bin_width', help='The bin width to use for correlations.', type=float, default=1.0)
parser.add_argument('-g', '--group', help='The quality of sorting for randomly chosen_cells.', default=['good', 'mua', 'unsorted'], type=str, nargs='*')
parser.add_argument('-p', '--probe', help='Filter the randomly chosen cells by probe', default=['posterior', 'frontal'], type=str, nargs='*')
parser.add_argument('-r', '--region', help='Filter the randomly chosen cells by region', default=['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1'], type=str, nargs='*')
parser.add_argument('-j', '--stim_id', help='A stim_id for use in the correlations vs bin length.', default=2, type=int)
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
parser.add_argument('-x', '--prefix', help='A prefix for the image file names.', type=str, default='')
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
image_dir = os.path.join(proj_dir, 'images')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
mat_dir = os.path.join(proj_dir, 'mat')

# loading useful functions
sys.path.append(py_dir)
import regionalCorrelations as rc
import regionalCorrelationsPlotting as rcp

def getPairwiseMeasurementFrame(pairs, exp_frame, cell_info, stim_id, bin_width):
    mutual_infos = np.zeros((pairs.shape[0], 4))
    correlation_coefficients = np.zeros(pairs.shape[0])
    p_values = np.zeros(pairs.shape[0])
    for i, pair in enumerate(pairs):
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

def getBiggestComponent(pairwise_measure_matrix):
    adjacency_matrix = (pairwise_measure_matrix > 0).astype(int)
    comp_assign, comp_size = bct.get_components(adjacency_matrix)
    keep_indices = np.nonzero(comp_assign == comp_size.argmax() + 1)[0]
    biggest_comp = pairwise_measure_matrix[keep_indices][:,keep_indices]
    np.fill_diagonal(biggest_comp, 0)
    return biggest_comp, keep_indices, comp_assign, comp_size

def checkDirected(network_matrix): # a symmetric matrix represents an undirected network
    if (network_matrix == network_matrix.T).all():
        return network_matrix
    else:
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Network is undirected! Converting to directed...')
        return (network_matrix + network_matrix.T)/2.0

def getUnifyingScaleCoef(pairwise_measure_matrix):
    return 1/pairwise_measure_matrix[pairwise_measure_matrix > 0].min()

def convertPairwiseMeasureMatrix(pairwise_measure_matrix, scale_coef = 100):
    pairwise_measure_int = pairwise_measure_matrix.astype(int)
    is_all_int = (pairwise_measure_matrix == pairwise_measure_int).all()
    if is_all_int:
        scale_coef = 1
    else:
        pairwise_measure_int = (scale_coef * pairwise_measure_matrix).round().astype(int)
    return pairwise_measure_int, scale_coef

def recoverPairwiseMeasureMatrix(pairwise_measure_int, scale_coef = 100):
    if scale_coef == 1:
        return pairwise_measure_int
    else:
        return pairwise_measure_int / scale_coef

def getExpectedNetworkFromData(pairwise_measure_int):
    return np.outer(np.sum(pairwise_measure_int, axis=0), np.sum(pairwise_measure_int, axis=1)) / np.sum(pairwise_measure_int).astype(float)

def getExpectedNetworkFromSamples(null_net_samples):
    num_samples = null_net_samples.shape[0]
    return null_net_samples.sum(axis=0)/num_samples

def getPoissonRates(expected_weights, strength_distn, weight_to_place, has_loops=False):
    poisson_rates = np.zeros(expected_weights.shape)
    expected_num_links = np.triu(expected_weights, k=int(not(has_loops)))
    pair_rows, pair_cols = np.nonzero(expected_num_links)
    prob_is_link = strength_distn[pair_rows] * strength_distn[pair_cols]
    prob_is_link = prob_is_link / prob_is_link.sum()
    poisson_rates[pair_rows, pair_cols] = weight_to_place * prob_is_link # the rate calculation and indexing all work perfectly, checked
    return poisson_rates

def sampleNullNetworkFullPoisson(poisson_rates, expected_net, scale_coef):
    sample_net = np.random.poisson(poisson_rates)
    sample_net = sample_net.T + sample_net # symmetrise
    sample_net = recoverPairwiseMeasureMatrix(sample_net, scale_coef)
    return sample_net

def sampleNullNetworkSparsePoisson(strength_distn, scale_coef, total_degrees, int_total_strength, total_weights, prob_link, has_loops):
    adjacency_sample = (np.random.rand(strength_distn.size, strength_distn.size) < np.triu(prob_link, k=1)).astype(int)
    pair_rows, pair_cols = np.nonzero(np.triu(adjacency_sample, k=0))
    num_links_to_place = int(round(int_total_strength/2.0)) - pair_rows.size # nLinks
    if (total_weights != total_degrees) & (num_links_to_place > 0): # we have a weighted network, with links to place
        poisson_rates = getPoissonRates(adjacency_sample, strength_distn, num_links_to_place, has_loops=has_loops)
        sampled_weights = np.random.poisson(poisson_rates)
        sample_net = sampled_weights + adjacency_sample
        sample_net = sample_net.T + sample_net
    else:
        sample_net = adjacency_sample.T + adjacency_sample
    return recoverPairwiseMeasureMatrix(sample_net, scale_coef)

def getDescSortedEigSpec(m):
    if (m == m.T).all(): # check symmetric
        eig_vals, eig_vecs = np.linalg.eigh(m)
    else:
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Input matrix is not symmetric...')
        eig_vals, eig_vecs = np.linalg.eig(m)
    desc_sort_inds = np.argsort(eig_vals)[::-1]
    return eig_vals[desc_sort_inds], eig_vecs[desc_sort_inds]

def getSparsePoissonWeightedConfModel(pairwise_measure_matrix, pairwise_measure_int, num_samples, expected_net, strength_distn, total_weights, scale_coef, has_loops, return_type, return_eig_vecs):
    num_nodes = pairwise_measure_matrix.shape[0]
    total_degrees = (pairwise_measure_matrix > 0).astype(int).sum() # K
    int_total_strength = pairwise_measure_int.sum()
    prob_link = getExpectedNetworkFromData(pairwise_measure_matrix > 0) # pnode, matches
    net_samples = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        net_samples[i] = sampleNullNetworkSparsePoisson(strength_distn, scale_coef, total_degrees, int_total_strength, total_weights, prob_link, has_loops=has_loops)
    expected = getExpectedNetworkFromSamples(net_samples) if return_type in ['expected', 'both'] else expected_net
    samples_eig_vals = np.zeros([num_samples, num_nodes])
    samples_eig_vecs = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        # samples_eig_vals[i], samples_eig_vecs[i] = getDescSortedEigSpec(net_samples[i] - expected)
        samples_eig_vals[i], samples_eig_vecs[i] = np.linalg.eigh(net_samples[i] - expected)
    if return_type == 'expected':
        optional_returns = {'expected_wcm':expected}
    elif return_type in 'all':
        optional_returns = {'expected_net':expected, 'net_samples':net_samples}
    elif return_type == 'both':
        optional_returns = {'expected_wcm':expected, 'net_samples':net_samples}
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unrecognised return type...')
    if return_eig_vecs:
        optional_returns['eig_vecs'] = samples_eig_vecs
    return samples_eig_vals, optional_returns

def getFullPoissonWeightedConfModel(num_samples, expected_net, strength_distn, total_weights, scale_coef, has_loops, return_type, return_eig_vecs):
    poisson_rates = getPoissonRates(expected_net, strength_distn, total_weights, has_loops=has_loops)
    num_nodes = expected_net.shape[0]
    net_samples = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        net_samples[i] = sampleNullNetworkFullPoisson(poisson_rates, expected_net, scale_coef)
    expected = getExpectedNetworkFromSamples(net_samples) if return_type == ['expected', 'both'] else expected_net
    samples_eig_vals = np.zeros([num_samples, num_nodes])
    samples_eig_vecs = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        # samples_eig_vals[i], samples_eig_vecs[i] = getDescSortedEigSpec(net_samples[i] - expected)
        samples_eig_vals[i], samples_eig_vecs[i] = np.linalg.eigh(net_samples[i] - expected)
    if return_type == 'expected':
        optional_returns = {'expected_wcm':expected}
    elif return_type in 'all':
        optional_returns = {'expected_net':expected, 'net_samples':net_samples}
    elif return_type == 'both':
        optional_returns = {'expected_wcm':expected, 'net_samples':net_samples}
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unrecognised return type...')
    if return_eig_vecs:
        optional_returns['eig_vecs'] = samples_eig_vecs
    return samples_eig_vals, optional_returns

def getPoissonWeightedConfModel(pairwise_measure_matrix, num_samples, is_sparse=False, has_loops=False, return_type='expected', return_eig_vecs=False):
    # return type can be 'expected', 'all', or 'both'
    pairwise_measure_matrix = checkDirected(pairwise_measure_matrix)
    assert (pairwise_measure_matrix >= 0).all(),dt.datetime.now().isoformat() + ' ERROR: ' + 'Weights must be non-negative...'
    strength_distn = pairwise_measure_matrix.sum(axis=0) # sA
    pairwise_measure_int, scale_coef = convertPairwiseMeasureMatrix(pairwise_measure_matrix, scale_coef = getUnifyingScaleCoef(pairwise_measure_matrix))
    expected_net = getExpectedNetworkFromData(pairwise_measure_int) # P, matches
    total_weights = (pairwise_measure_int.sum()/2).astype(int) # nLinks
    if is_sparse:
        samples_eig_vals, optional_returns = getSparsePoissonWeightedConfModel(pairwise_measure_matrix, pairwise_measure_int, num_samples, expected_net, strength_distn, total_weights, scale_coef, has_loops, return_type, return_eig_vecs)
    else:
        samples_eig_vals, optional_returns = getFullPoissonWeightedConfModel(num_samples, expected_net, strength_distn, total_weights, scale_coef, has_loops, return_type, return_eig_vecs)
    return samples_eig_vals, optional_returns

def getConfidenceIntervalFromStdErr(st_dev, num_samples, interval):
    if interval == 0:
        if np.isscalar(st_dev):
            return 0.0
        else:
            return np.repeat(0.0, st_dev.shape)
    else:
        symm_interval = 1-(1-interval)/2.0
        t_val = stats.t.ppf(symm_interval, num_samples)
        return t_val * st_dev / np.sqrt(num_samples)

def getNonParaPredictionInterval(sample):
    num_data_points = sample.size
    cutoff = int(num_data_points/2.0)
    ordered_sample = np.sort(sample)
    prediction_intervals = np.zeros([cutoff, 3])
    for i in range(cutoff):
        pi = (1 - 2*(i+1) / (num_data_points + 1.0))
        prediction_intervals[i,:] = [pi, ordered_sample[i], ordered_sample[num_data_points - 1 - i]]
    return prediction_intervals

def getLowDimSpace(modularity_matrix, eig_vals, confidence_level, int_type='CI'):
    assert modularity_matrix.shape[0] == eig_vals.shape[1], dt.datetime.now().isoformat() + ' ERROR: ' + 'Eigenvalue matrix is the wrong shape...'
    # mod_eig_vals, mod_eig_vecs = getDescSortedEigSpec(modularity_matrix)
    mod_eig_vals, mod_eig_vecs = np.linalg.eigh(modularity_matrix)
    mins_eig, maxs_eig = eig_vals.min(axis=1), eig_vals.max(axis=1)
    if int_type == 'CI':
        mean_mins_eig = mins_eig.mean()
        min_confidence_ints = getConfidenceIntervalFromStdErr(mins_eig.std(), eig_vals.shape[0], confidence_level)
        eig_lower_confidence_int = mean_mins_eig - min_confidence_ints
        mean_maxs_eig = maxs_eig.mean()
        max_confidence_ints = getConfidenceIntervalFromStdErr(maxs_eig.std(), eig_vals.shape[0], confidence_level)
        eig_upper_confidence_int = mean_maxs_eig + max_confidence_ints
    elif int_type == 'PI':
        if 1 == np.mod(maxs_eig.size, 2):
            prediction_intervals_min, prediction_intervals_max = getNonParaPredictionInterval(mins_eig), getNonParaPredictionInterval(maxs_eig)
        else:
            prediction_intervals_min, prediction_intervals_max = getNonParaPredictionInterval(mins_eig[0:-1]), getNonParaPredictionInterval(maxs_eig[0:-1])
        try:
            mean_mins_eig, min_confidence_ints = prediction_intervals_min[prediction_intervals_min[:,0] == confidence_level, 1:3][0]
            mean_maxs_eig, max_confidence_ints = prediction_intervals_max[prediction_intervals_max[:,0] == confidence_level, 1:3][0]
            eig_upper_confidence_int = prediction_intervals_max[prediction_intervals_max[:,0] == confidence_level, 2][0]
            eig_lower_confidence_int = prediction_intervals_min[prediction_intervals_min[:,0] == confidence_level, 1][0]
        except:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Cannot find specified prediction interval!')
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown interval type!')
    exceeding_upper_bound_inds = np.flatnonzero(mod_eig_vals >= eig_upper_confidence_int)
    below_lower_bound_inds = np.flatnonzero(mod_eig_vals <= eig_lower_confidence_int)
    exceeding_eig_space = mod_eig_vecs[:,exceeding_upper_bound_inds]
    below_eig_space = mod_eig_vecs[:,below_lower_bound_inds]
    return below_eig_space, below_lower_bound_inds, [mean_mins_eig, min_confidence_ints], exceeding_eig_space, exceeding_upper_bound_inds, [mean_maxs_eig, max_confidence_ints]

def nodeRejection(modularity_matrix, eig_vals, confidence_level, eig_vecs, weight_type='linear', norm='L2', int_type='CI', bounds='upper'):

    def getBoundedSpace(bounds, modularity_matrix, eig_vals, confidence_level, int_type):
        upper_and_lower = getLowDimSpace(modularity_matrix, eig_vals, confidence_level, int_type=int_type)
        lowd_eig_space, lowd_indices = upper_and_lower[3:5] if bounds == 'upper' else upper_and_lower[0:2]
        if not(bounds in ['upper', 'lower']):
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown bound!')
        return lowd_eig_space, lowd_indices

    def getWeightedLowDSpaceProjections(weight_type, num_samples, lowd_eig_space, lowd_indices, eig_vals, eig_vecs, mod_eig_vals):
        if weight_type == 'none':
            weighted_lowd_space = lowd_eig_space
            weighted_model_projections = eig_vecs[:,:,lowd_indices]
        elif weight_type == 'linear':
            weighted_lowd_space = mod_eig_vals[lowd_indices] * lowd_eig_space # Vweighted
            weighted_model_projections = np.array([eig_vals[i, lowd_indices] * eig_vecs[i,:,lowd_indices].T for i in range(num_samples)]) # VmodelW
        elif weight_type == 'sqrt':
            weighted_lowd_space = np.sqrt(mod_eig_vals[lowd_indices]) * lowd_eig_space
            weighted_model_projections = np.array([np.sqrt(eig_vals[i, lowd_indices]) * eig_vecs[i,:,lowd_indices].T for i in range(num_samples)])
        else:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown weight type!')
        return weighted_lowd_space, weighted_model_projections

    def getLowDModelLengths(norm, weighted_lowd_space, weighted_model_projections):
        if norm == 'L2':
            lowd_lengths = np.sqrt(np.power(weighted_lowd_space, 2).sum(axis=1))
            model_lengths = np.sqrt(np.power(weighted_model_projections,2).sum(axis=2)) # VmodelL
        elif norm == 'L1':
            lowd_lengths = np.abs(weighted_lowd_space).sum(axis=1)
            model_lengths = np.abs(weighted_model_projections).sum(axis=2)
        elif norm == 'Lmax':
            lowd_lengths = np.abs(weighted_lowd_space).max(axis=1)
            model_lengths = np.abs(weighted_model_projections).max(axis=2)
        else:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown norm!')
        return lowd_lengths, model_lengths

    num_samples, num_nodes = eig_vals.shape
    # mod_eig_vals = getDescSortedEigSpec(modularity_matrix)[0]
    mod_eig_vals = np.linalg.eigh(modularity_matrix)[0]
    lowd_eig_space, lowd_indices = getBoundedSpace(bounds, modularity_matrix, eig_vals, confidence_level, int_type)
    weighted_lowd_space, weighted_model_projections = getWeightedLowDSpaceProjections(weight_type, num_samples, lowd_eig_space, lowd_indices, eig_vals, eig_vecs, mod_eig_vals)
    lowd_lengths, model_lengths = getLowDModelLengths(norm, weighted_lowd_space, weighted_model_projections)
    reject_dict = {}
    reject_dict['m_model'] = model_lengths.mean(axis=0)
    if int_type == 'CI':
        reject_dict['CI_model'] = getConfidenceIntervalFromStdErr(model_lengths.std(axis=0), num_samples, confidence_level)
        reject_dict['difference'] = {}; reject_dict['neg_difference'] = {}
        reject_dict['difference']['raw'] = lowd_lengths - (reject_dict['m_model'] + reject_dict['CI_model'])
        reject_dict['difference']['norm'] = reject_dict['difference']['raw'] / (reject_dict['m_model'] + reject_dict['CI_model'])
        reject_dict['neg_difference']['raw'] = lowd_lengths - (reject_dict['m_model'] - reject_dict['CI_model'])
        reject_dict['neg_difference']['norm'] = reject_dict['neg_difference']['raw'] / (reject_dict['m_model'] - reject_dict['CI_model'])
    elif int_type == 'PI':
        reject_dict['PI_model'] = np.zeros([num_nodes, 2])
        for i in range(num_nodes):
            prediction_intervals = getNonParaPredictionInterval(model_lengths[:,i]) if 1 == np.mod(model_lengths.shape[0], 2) else getNonParaPredictionInterval(model_lengths[0:-1,i])
            try:
                reject_dict['PI_model'][i,:] = prediction_intervals[prediction_intervals[:,0] == confidence_level, 1:3][0]
            except:
                sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Cannot find specified prediction interval!')
        reject_dict['difference'] = {}; reject_dict['neg_difference'] = {}
        reject_dict['difference']['raw'] = lowd_lengths - reject_dict['PI_model'][:,1]
        reject_dict['difference']['norm'] = reject_dict['difference']['raw'] / reject_dict['PI_model'][:,1]
        reject_dict['neg_difference']['raw'] = lowd_lengths - reject_dict['PI_model'][:,0]
        reject_dict['neg_difference']['raw'] = lowd_lengths / reject_dict['PI_model'][:,0]
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown interval type!')
    reject_dict['signal_inds'] = np.flatnonzero(reject_dict['difference']['raw'] > 0)
    reject_dict['noise_inds'] = np.flatnonzero(reject_dict['difference']['raw'] <= 0)
    reject_dict['neg_signal_inds'] = np.flatnonzero(reject_dict['neg_difference']['raw'] <= 0)
    return reject_dict

def kMeansSweep(e_vectors, min_groups, max_groups, reps, dims):
    assert min_groups >= 2, dt.datetime.now().isoformat() + ' ERROR: ' + 'Minimum number of groups must be at least 2!'
    assert min_groups <= max_groups, dt.datetime.now().isoformat() + ' ERROR: ' + 'Minimum number of groups greater than maximum number of groups!'
    num_nodes, num_e_vectors = e_vectors.shape
    if (dims == 'scale') & (num_e_vectors < max_groups - 1):
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Not enough embedding dimensions to scale upper bound!')
    num_possible_groups = 1 + max_groups - min_groups
    clusterings = np.zeros([num_nodes, reps], dtype=int)
    for num_groups in range(min_groups, max_groups + 1):
        if dims == 'all':
            this_vector = e_vectors
        elif dims == 'scale':
            this_vector = e_vectors[:, num_groups-1]
        else:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown dims!')
        for i in range(reps):
            kmeans_estimator = KMeans(init='k-means++', n_clusters=num_groups, n_init=10)
            kmeans_estimator.fit(this_vector)
            clusterings[:,i] = kmeans_estimator.labels_ # each column is a clustering
    return clusterings

def getClusteringModularity(clustering, modularity_matrix, m=1):
    num_nodes = clustering.size
    num_clusters = clustering.max() + 1
    membership_matrix = np.zeros([num_nodes, num_clusters], dtype=int)
    for k in range(num_nodes):
        membership_matrix[k, clustering[k]] = 1
    modularity = np.matrix.trace(np.dot(np.dot(membership_matrix.T, modularity_matrix), membership_matrix)) / (2*m)
    return modularity

def checkConvergenceConsensus(consensus_matrix):
    num_nodes = consensus_matrix.shape[0]
    pair_rows, pair_cols = np.array(list(combinations(range(num_nodes),2))).T # upper triangle indices
    consensuses = consensus_matrix[pair_rows, pair_cols] # order differs from Matlab
    bin_width = 0.01 # Otsu's method
    if (consensuses == 1).all():
        threshold = -np.inf
    else:
        bin_counts, edges = np.histogram(consensuses, bins = np.arange(consensuses.min(), consensuses.max(), bin_width))
        threshold_bin = threshold_otsu(bin_counts)
        threshold = edges[threshold_bin]
    high_consensus = consensus_matrix.copy()
    high_consensus[consensus_matrix < threshold] = 0
    consensus_clustering = np.array([], dtype=int)
    clustered_nodes = np.array([], dtype=int)
    is_converged = False
    consensus_clustering_iterations = 0
    for i in range(num_nodes):
        if not(i in clustered_nodes): # if not already sorted
            this_cluster = np.hstack([i, np.flatnonzero(high_consensus[i,:] > 0)])
            # if any of the nodes in this cluster are already in a cluster, then this is not transitive
            if np.intersect1d(clustered_nodes, this_cluster).size > 0:
                is_converged = False
                return is_converged, consensus_clustering, threshold
            else:
                is_converged = True
            consensus_clustering = np.hstack([consensus_clustering, consensus_clustering_iterations * np.ones(this_cluster.size, dtype=int)])
            clustered_nodes = np.hstack([clustered_nodes, this_cluster])
            consensus_clustering_iterations += 1
    sort_inds = np.argsort(clustered_nodes)
    return is_converged, consensus_clustering[sort_inds], threshold

def nullModelConsensusSweep(cluster_sizes, num_allowed_clusterings, num_nodes):
    assert cluster_sizes.size == num_allowed_clusterings.size, dt.datetime.now().isoformat() + ' ERROR: ' + 'number of cluster sizes != number of allowed clusterings'
    num_draws = np.zeros(cluster_sizes.shape)
    draw_var = np.zeros(cluster_sizes.shape)
    for i, (cluster_size, num_allowed) in enumerate(zip(cluster_sizes, num_allowed_clusterings)):
        uniform_same_cluster_prob = 1/float(cluster_size)
        num_draws[i] = num_allowed * uniform_same_cluster_prob
        draw_var[i] = uniform_same_cluster_prob * num_allowed * (1 - uniform_same_cluster_prob)
    exp_proportions = np.zeros([num_nodes, num_nodes]) + num_draws.sum()/num_allowed_clusterings.sum()
    proportion_var = np.zeros([num_nodes, num_nodes]) + draw_var.sum()/num_allowed_clusterings.sum()
    np.fill_diagonal(exp_proportions, 0.0)
    np.fill_diagonal(proportion_var, 0.0)
    return exp_proportions, proportion_var

def embedConsensusNull(consensus_matrix, consensus_type, cluster_sizes, num_allowed_clusterings):
    num_nodes = consensus_matrix.shape[0]
    if consensus_type == 'sweep':
        exp_proportions, proportion_var = nullModelConsensusSweep(cluster_sizes, num_allowed_clusterings, num_nodes)
    elif consensus_type == 'expect':
        return 0 # implement this
    else:
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unknown consensus type!')
    cons_mod_matrix = consensus_matrix - exp_proportions
    cons_eig_vals, cons_eig_vecs = np.linalg.eigh(cons_mod_matrix)
    low_d_consensus = cons_eig_vecs[:,cons_eig_vals > 0]
    est_num_groups = (cons_eig_vals > 0).sum() + 1
    return low_d_consensus, cons_mod_matrix, est_num_groups, cons_eig_vals

def consensusCommunityDetect(signal_measure_matrix, signal_expected_wcm, min_groups, max_groups, kmeans_reps=100, dims='all', is_explore=True):
    if (np.diag(signal_measure_matrix) != 0).any():
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Measure matrix has self loops...')
    if (signal_measure_matrix < 0).any():
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Measure matrix has negative values...')
    num_nodes = signal_measure_matrix.shape[0]
    total_unique_weight = signal_measure_matrix.sum()/2.0
    consensus_iterations = 1
    is_converged = False
    modularity_matrix = signal_measure_matrix - signal_expected_wcm
    # mod_eig_vals, mod_eig_vecs = getDescSortedEigSpec(modularity_matrix)
    mod_eig_vals, mod_eig_vecs = np.linalg.eigh(modularity_matrix)
    e_vectors = mod_eig_vecs[:,1-max_groups:]
    kmeans_clusterings = kMeansSweep(e_vectors, min_groups, max_groups, kmeans_reps, dims) # C, can't get the clusterings to match
    clustering_modularities = np.array([getClusteringModularity(clustering, modularity_matrix, total_unique_weight) for clustering in kmeans_clusterings.T]) # Q
    if (kmeans_clusterings == 0).all() | (clustering_modularities <= 0).all():
        return 0 # return empty results
    max_modularity = clustering_modularities.max()
    max_mod_cluster = kmeans_clusterings[:,clustering_modularities.argmax()]
    while not(is_converged):
        allowed_clusterings = kmeans_clusterings[:,clustering_modularities > 0]
        consensus_matrix = bct.agreement(allowed_clusterings) / float(kmeans_reps)
        is_converged, consensus_clustering, threshold = checkConvergenceConsensus(consensus_matrix) # doesn't match. Some investigation required.
        if is_converged:
            consensus_modularity = getClusteringModularity(consensus_clustering, modularity_matrix, total_unique_weight)
        else:
            consensus_iterations += 1
            if consensus_iterations > 50:
                print(dt.datetime.now().isoformat() + ' WARN: ' + 'Not converged after 50 reps. Exiting...')
                consensus_clustering = np.array([]); consensus_modularity = 0.0;
                return max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations
            else:
                if (min_groups == max_groups) & (not(is_explore)):
                    num_allowed_clusterings = np.array([allowed_clusterings.shape[1]])
                    low_d_consensus, cons_mod_matrix, est_num_groups, cons_eig_vals = embedConsensusNull(consensus_matrix, 'sweep', np.array(min_groups, max_groups+1), num_allowed_clusterings)
                    if est_num_groups >= max_groups:
                        kmeans_clusterings = kMeansSweep(low_d_consensus[:,1-max_groups:], min_groups, max_groups, kmeans_reps, dims)
                    elif (low_d_consensus == 0).all():
                        kmeans_clusterings = np.array([])
                    else:
                        kmeans_clusterings = kMeansSweep(low_d_consensus, min_groups, max_groups, kmeans_reps, dims)
                if (min_groups != max_groups) | is_explore:
                    num_allowed_clusterings = (clustering_modularities>0).reshape([kmeans_reps, 1+ max_groups - min_groups]).sum(axis=0)
                    low_d_consensus, cons_mod_matrix, est_num_groups, cons_eig_vals = embedConsensusNull(consensus_matrix, 'sweep', np.arange(min_groups, max_groups + 1), num_allowed_clusterings)
                    max_groups = est_num_groups
                    if max_groups < min_groups: min_groups = max_groups
                    kmeans_clusterings = kMeansSweep(low_d_consensus, min_groups, max_groups, kmeans_reps, dims)
                if (kmeans_clusterings == 0.0).all():
                    print(dt.datetime.now().isoformat() + ' WARN: ' + 'Consensus matrix projection is empty. Exiting...')
                    consensus_clustering = np.array([]); consensus_modularity = 0.0;
                    return max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations
                else:
                    clustering_modularities = np.array([getClusteringModularity(clustering, modularity_matrix, total_unique_weight) for clustering in kmeans_clusterings.T])
    return max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations


print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading trials info...')
trials_info = rc.getStimTimesIds(stim_info, args.stim_id)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Selecting cells...')
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
# now we have the measurements,
# we want to sample from the null space
# then we want to get all the eigenvalues
# then we want to get the smaller subspace (if it exists)
# then we want to cluster that
# then we want to show the clusters

info_matrix, keep_indices, comp_assign, comp_size = getBiggestComponent(info_matrix)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling from null model...')
samples_eig_vals, null_net_samples = getPoissonWeightedConfModel(info_matrix, 100)

# testing with Les Miserables data
# node rejection here:
m = loadmat(os.path.join(os.environ['PROJ'], 'Network_Noise_Rejection', 'Networks', 'lesmis.mat'))
problem = m['Problem']
A = problem['A'][0][0].todense().A
data_keys = ['A', 'ixRetain', 'Comps', 'CompSizes', 'Emodel', 'ExpA', 'B']
data = {data_keys[i]:v for i,v in enumerate(getBiggestComponent(A))} # all match
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting E-values and <P> of sparse WCM...')
wcm = getPoissonWeightedConfModel(data['A'], 100, is_sparse=True, return_eig_vecs=True)
data['Emodel'] = wcm[0]; data['ExpA'] = wcm[1]['expected_wcm'];Vmodel = wcm[1]['eig_vecs']
data['B'] = data['A'] - data['ExpA']
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Rejecting nodes...')
reject_dict = nodeRejection(data['B'], data['Emodel'], 0, Vmodel, weight_type='linear', norm='L2', int_type='CI', bounds='upper') # extra signal nodes compared to Matlab version
data['Asignal'] = data['A'][reject_dict['signal_inds']][:,reject_dict['signal_inds']]
new_keys = ['Asignal_comp', 'signal_ixRetain', 'SignalComps', 'SignalComp_sizes']
data.update({new_keys[i]:v for i,v in enumerate(getBiggestComponent(data['Asignal']))})
data['ixSignal_comp'] = reject_dict['signal_inds'][data['signal_ixRetain']]
K = data['Asignal_comp'].sum(axis=0)
ixLeaves = np.flatnonzero(K==1)
ixKeep = np.flatnonzero(K > 1)
data['ixSignal_Final'] = data['ixSignal_comp'][ixKeep]
data['ixSignal_Leaves'] = data['ixSignal_comp'][ixLeaves]
data['Asignal_final'] = data['Asignal_comp'][ixKeep][:,ixKeep]

# commenting out low d space lines to work on node rejection
# below_eig_space, below_lower_bound_inds, [mean_mins_eig, min_confidence_ints], exceeding_eig_space, exceeding_upper_bound_inds, [mean_maxs_eig, max_confidence_ints] = getLowDimSpace(data['B'], data['Emodel'], 0)
# data['Dspace'] = exceeding_eig_space; data['Dn'] = exceeding_eig_space.shape[1]; data['EigEst'] = [mean_maxs_eig, max_confidence_ints];
# data['Nspace'] = below_eig_space; data['Dneg'] = below_eig_space.shape[1]; data['NEigEst'] = [mean_mins_eig, min_confidence_ints];

# clustering part here:
r = loadmat(os.path.join(os.environ['PROJ'], 'Network_Noise_Rejection', 'Results', 'Rejected_Lesmis.mat'))
Data = r['Data'][0,0];
nodelabels = Data['nodelabels']
pyData = {}; pyData['Dn'] = Data['Dn'][0][0]; pyData['ixSignal_Final'] = Data['ixSignal_Final'].flatten().astype(int)-1;
pyData['ExpA'] = Data['ExpA']; pyData['Asignal_final'] = Data['Asignal_final'];
(pyData['Dn'] > 0) & (pyData['ixSignal_Final'].size > 3)
P = pyData['ExpA'][pyData['ixSignal_Final']][:, pyData['ixSignal_Final']]
signal_measure_matrix, signal_expected_wcm, min_groups, max_groups = pyData['Asignal_final'], P, pyData['Dn'] + 1, pyData['Dn'] + 1
max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations = consensusCommunityDetect(signal_measure_matrix, signal_expected_wcm, min_groups, max_groups)
