import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import STCDT as cd
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
proj_dir = os.path.join(os.environ['HOME'], 'Regional_Correlations')
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
    corr_matrix = np.zeros([num_cells, num_cells]); symm_unc_matrix = np.zeros([num_cells, num_cells]);
    for pair in pairs:
        first_index = region_sorted_cell_ids.tolist().index(pair[0])
        second_index = region_sorted_cell_ids.tolist().index(pair[1])
        pair_record = pairwise_measurements[(pairwise_measurements.first_cell_id == pair[0]) & (pairwise_measurements.second_cell_id == pair[1])].iloc[0]
        corr_matrix[first_index, second_index] = corr_matrix[second_index, first_index] = pair_record.corr_coef
        symm_unc_matrix[first_index, second_index] = symm_unc_matrix[second_index, first_index] = pair_record.symm_unc_qe
    return corr_matrix, symm_unc_matrix

def getClusterAndMod(num_clusters, e_vectors, modularity_matrix):
    num_nodes, _ = e_vectors.shape
    init_centres = cd.initialise_centres(e_vectors, num_clusters)
    whitened_vectors = whiten(e_vectors)
    test_centres, labels = kmeans2(whitened_vectors, init_centres)
    membership_matrix = np.zeros([num_nodes, num_clusters], dtype=int)
    for k in range(num_nodes):
        membership_matrix[k, labels[k]] = 1
    modularity = np.matrix.trace(np.dot(np.dot(membership_matrix.T, modularity_matrix), membership_matrix))
    return labels, modularity

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
    return clustering_labels

def getManyClusterings(pairwise_measure_matrix, num_clusterings=50):
    num_nodes = pairwise_measure_matrix.shape[0]
    null_network = np.outer(np.sum(pairwise_measure_matrix, axis=0), np.sum(pairwise_measure_matrix, axis=1)) / np.sum(pairwise_measure_matrix)
    modularity_matrix = pairwise_measure_matrix - null_network
    eigenvalues, eigenvectors = np.linalg.eigh(modularity_matrix)
    e_vectors = eigenvectors[:, eigenvalues>0] 
    clusterings = np.vstack([kMeansSweep(e_vectors, modularity_matrix) for i in range(num_clusterings)])
    return clusterings

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
corr_matrix, symm_unc_matrix = getPairwiseMeasurementMatrices(pairs, region_sorted_cell_ids, pairwise_measurements)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting many clusterings...')
many_clusterings = getManyClusterings(symm_unc_matrix)
