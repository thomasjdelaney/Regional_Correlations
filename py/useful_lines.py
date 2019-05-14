import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import STCDT as cd
import bct
from scipy.io import loadmat
from itertools import combinations
from scipy.cluster.vq import whiten, kmeans2

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

def getClusteringModularity(clustering, modularity_matrix):
    num_nodes = clustering.size
    num_clusters = clustering.max() + 1
    membership_matrix = np.zeros([num_nodes, num_clusters], dtype=int)
    for k in range(num_nodes):
        membership_matrix[k, clustering[k]] = 1
    modularity = np.matrix.trace(np.dot(np.dot(membership_matrix.T, modularity_matrix), membership_matrix))
    return modularity

def getClusterAndMod(num_clusters, e_vectors, modularity_matrix):
    init_centres = cd.initialise_centres(e_vectors, num_clusters)
    whitened_vectors = whiten(e_vectors)
    test_centres, clustering = kmeans2(whitened_vectors, init_centres)
    return clustering, getClusteringModularity(clustering, modularity_matrix)

def kMeansSweep(e_vectors, modularity_matrix, reps=7):
    num_nodes, num_e_vectors = e_vectors.shape
    max_num_clusters = num_e_vectors + 1
    clustering_labels = np.zeros([num_e_vectors, num_nodes])
    Q = 0 # modularity, we want to maximise this
    for num_clusters in range(2, max_num_clusters + 1):
        for j in range(reps):
            labels, modularity = getClusterAndMod(num_clusters, e_vectors, modularity_matrix)
            if modularity > Q:
                clustering_labels, Q  = labels, modularity
    return clustering_labels, Q

def getManyClusteringsWithMods(pairwise_measure_matrix, num_clusterings=100):
    num_nodes = pairwise_measure_matrix.shape[0]
    null_network = np.outer(np.sum(pairwise_measure_matrix, axis=0), np.sum(pairwise_measure_matrix, axis=1)) / np.sum(pairwise_measure_matrix)
    modularity_matrix = pairwise_measure_matrix - null_network
    eigenvalues, eigenvectors = np.linalg.eigh(modularity_matrix)
    e_vectors = eigenvectors[:, eigenvalues>0]
    clusterings = np.zeros([num_clusterings, num_nodes], dtype=int)
    modularities = np.zeros(num_clusterings)
    for i in range(num_clusterings):
        clusterings[i], modularities[i] = kMeansSweep(e_vectors, modularity_matrix)
    return clusterings, modularities

def getBiggestComponent(pairwise_measure_matrix):
    comp_assign, comp_size = bct.get_components(pairwise_measure_matrix)
    keep_indices = np.nonzero(comp_assign == comp_size.argmax() + 1)[0]
    biggest_comp = pairwise_measure_matrix[keep_indices]
    np.fill_diagonal(biggest_comp, 0)
    return biggest_comp, keep_indices, comp_assign, comp_size

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

def getExpectedNullNetwork(pairwise_measure_int):
    return np.outer(np.sum(pairwise_measure_int, axis=0), np.sum(pairwise_measure_int, axis=1)) / np.sum(pairwise_measure_int).astype(float)

def sampleNullNetworkFullPoisson(num_nodes, expected_null, strength_distn, total_weights, pairwise_measure_matrix, scale_coef):
    sample_null_net = np.zeros([num_nodes, num_nodes])
    expected_num_links = np.triu(expected_null, k=1)
    pair_rows, pair_cols = np.nonzero(expected_num_links)
    prob_is_link = strength_distn[pair_rows] * strength_distn[pair_cols]
    prob_is_link = prob_is_link / prob_is_link.sum()
    poisson_rate = total_weights * prob_is_link
    sampled_num_links = np.random.poisson(poisson_rate)
    sample_null_net[pair_rows, pair_cols] = sampled_num_links
    sample_null_net = sample_null_net.T + sample_null_net # symmetrise
    sample_null_net = recoverPairwiseMeasureMatrix(sample_null_net, scale_coef)
    sample_eig_vals, sample_eig_vecs = np.linalg.eig(pairwise_measure_matrix - sample_null_net)
    sort_indices = np.argsort(sample_eig_vals)
    return sample_eig_vals[sort_indices], sample_null_net

def getPoissonWeightedConfModel(pairwise_measure_matrix, num_samples):
    num_nodes = pairwise_measure_matrix.shape[0]
    strength_distn = pairwise_measure_matrix.sum(axis=0)
    degree_distn = (pairwise_measure_matrix > 0).astype(int).sum(axis=0)
    pairwise_measure_int, scale_coef = convertPairwiseMeasureMatrix(pairwise_measure_matrix)
    total_weights = (pairwise_measure_int.sum()/2).astype(int) # nLinks
    expected_null = getExpectedNullNetwork(pairwise_measure_int)
    samples_eig_vals = np.zeros([num_samples, num_nodes])
    null_net_samples = np.zeros([num_samples, num_nodes, num_nodes])
    for i in range(num_samples):
        samples_eig_vals[i], null_net_samples[i] = sampleNullNetworkFullPoisson(num_nodes, expected_null, strength_distn, total_weights, pairwise_measure_int, scale_coef)
    return samples_eig_vals, null_net_samples

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
m = loadmat(os.path.join(os.environ['PROJ'], 'Network_Noise_Rejection', 'Networks', 'lesmis.mat'))
problem = m['Problem']
A = problem['A'][0][0].todense().A
data_keys = ['A', 'ixRetain', 'Comps', 'CompSizes', 'Emodel', 'ExpA']
data = {data_keys[i]:v for i,v in enumerate(getBiggestComponent(A))} # all match
data.update({data_keys[i+len(data)]:v for i,v in enumerate(getPoissonWeightedConfModel(data['A'], 100))})

# set conversion parameter
# print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting many clusterings...')
# clusterings, modularities = getManyClusteringsWithMods(info_matrix)
# agreement = bct.agreement(clusterings.T)/clusterings.shape[0]
#       Calculate consensus partition of agreement/consensus matrix
# NB picking tau may be a pain, cycle from 0.02 to 0.4 in steps of 0.2, calculate the modularity along the way.
