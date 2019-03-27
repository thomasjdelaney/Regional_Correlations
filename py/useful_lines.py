import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.io import loadmat
from itertools import product, combinations
from scipy.stats import pearsonr

np.random.seed(1798) # setting seed
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

def getIndexPairsWithDelay(num_samples, num_bins, delay):
    index_pairs = np.array([])
    for i,j in product(np.arange(num_bins), np.arange(num_bins)):
        if i-j == -delay:
            index_pairs = np.append(index_pairs,[i,j])
    num_pairs = index_pairs.size//2
    index_pairs = np.reshape(index_pairs, [num_pairs, 2]).astype(int)
    return num_pairs, index_pairs

def getCorrelationWithDelay(y, x, delay):
    num_samples, num_bins = y.shape
    num_pairs, index_pairs = getIndexPairsWithDelay(num_samples, num_bins, delay)
    correlations = np.zeros(num_pairs)
    p_values = np.zeros(num_pairs)
    for ind, (i, j) in enumerate(index_pairs):
        first = y[:,i]
        second = x[:,j]
        if any(first) & any(second):
            correlations[ind], p_values[ind] = pearsonr(first, second)
    non_zero_inds = correlations.nonzero()[0]
    corr_coef_estimate = correlations[correlations.nonzero()].mean()
    corr_coef_stderr = correlations[correlations.nonzero()].std()/np.sqrt(non_zero_inds.size)
    return corr_coef_estimate, corr_coef_stderr

def plotWithStdErrors(measure, std_errors, delays, ylabel):
    plt.plot(delays, measure)
    plt.fill_between(delays, measure - std_errors, measure + std_errors, color='blue', alpha=0.3, label='standard error')
    plt.ylabel(ylabel, fontsize='large')
    plt.xlabel('Delay (time bins)', fontsize='large')
    plt.xlim([delays.min(), delays.max()])
    plt.legend(fontsize='large')

cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
cell_ids = cell_info[(cell_info.region=='thalamus')&(cell_info.group=='good')].index.values
trials_info = rc.getStimTimesIds(stim_info, 2)
num_trials = trials_info.shape[0]
spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
responding_pairs = rc.getRespondingPairs(cell_ids, trials_info, spike_time_dict, cell_info, 5, is_strong=True)
responding_cells = np.unique(responding_pairs)
exp_frame = rc.getExperimentFrame(responding_cells, trials_info, spike_time_dict, cell_info, 0.001)
pair = responding_pairs[0]
first_response = exp_frame[exp_frame.cell_id == pair[0]]['num_spikes']
second_response = exp_frame[exp_frame.cell_id == pair[1]]['num_spikes']
num_responses = first_response.size
num_bins = num_responses//num_trials
first_trials = first_response.values.reshape(num_trials, num_bins)
second_trials = second_response.values.reshape(num_trials, num_bins)
delays = np.arange(-50, 51)
correlations = np.zeros(delays.shape)
corr_std_errors = np.zeros(delays.shape)
for i, delay in enumerate(delays):
    correlations[i], corr_std_errors[i] = getCorrelationWithDelay(first_trials, second_trials, delay)
plotWithStdErrors(correlations, corr_std_errors, delays, "Correlation Coefficient (a.u.)")
plt.show(block=False)
