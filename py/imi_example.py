"""
For doing examples of calculating the incremental mutual information. Experiments are supposed to match those in the paper
'Incremental Mutual Information: A New Method for Characterizing the Strength and Dynamics of Connections in Neuronal Circuits'
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001035#s2
"""
import sys, argparse
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.random import normal
from scipy.stats.stats import pearsonr
from itertools import product

parser = argparse.ArgumentParser(description='Reproducing some results from the IMI paper.')
parser.add_argument('-v', '--underlying_variance', help='The variance of the underlying normal distributions.', type=float, default=1.0)
parser.add_argument('-e', '--epsilon', help='The strength of the connection.', type=float, default=0.5)
parser.add_argument('-n', '--num_samples', help='The number of samples to make.', type=int, default=1000)
parser.add_argument('-p', '--peak_delay', help='The delay for the effect of y on x.', type=int, default=4)
parser.add_argument('-b', '--num_bins', help='The number of bins to use in each sample', type=int, default=30)
args = parser.parse_args()

def getFilteredSpikeTrains(trains, filter_var=1, filter_window=3):
    num_trials, num_bins = trains.shape
    trial_inds, spike_inds = np.where(trains == 1)
    filter_without_ends = filtered_y = gaussian_filter1d(trains, filter_var, mode='constant', cval=0.0)
    filtered_trains = np.zeros(trains.shape)
    for i in np.arange(num_trials):
        trial_spike_inds = spike_inds[trial_inds == i]
        if trial_spike_inds.size == 0: continue
        filter_inds = np.concatenate([ts + np.arange(-filter_window,filter_window+1)for ts in trial_spike_inds])
        filter_inds = filter_inds[(filter_inds >= 0)&(filter_inds < num_bins)]
        filtered_trains[i,filter_inds] = filter_without_ends[i,filter_inds]
    return filtered_trains

def getCorrelatedSamples(num_samples, num_bins, underlying_variance, peak_delay, epsilon):
    s_x = normal(0,underlying_variance,[num_samples,num_bins])
    s_y = normal(0,underlying_variance,[num_samples,num_bins])
    a_y = np.min([np.ones(s_y.shape), np.max([np.zeros(s_y.shape),np.floor(s_y)], axis=0)], axis=0)
    filtered_y = getFilteredSpikeTrains(a_y)
    under_x = s_x + epsilon*filtered_y
    a_x = np.min([np.ones(s_x.shape), np.max([np.zeros(s_x.shape), np.floor(under_x)], axis=0)], axis=0)
    return a_y, np.roll(a_x, peak_delay, axis=1)

def getCorrelationWithDelay(y, x, delay):
    num_samples, num_bins = y.shape
    index_pairs = np.array([])
    for i,j in product(np.arange(num_bins), np.arange(num_bins)):
        if i-j == -delay:
            index_pairs = np.append(index_pairs,[i,j])
    num_pairs = index_pairs.size//2
    index_pairs = np.reshape(index_pairs, [num_pairs, 2]).astype(int)
    correlations = np.zeros(num_pairs)
    p_values = np.zeros(num_pairs)
    for ind, (i, j) in enumerate(index_pairs):
        correlations[ind], p_values[ind] = pearsonr(y[:,i], x[:,j])
    return correlations.mean(), correlations.std()/np.sqrt(num_pairs)

y, x = getCorrelatedSamples(args.num_samples, args.num_bins, args.underlying_variance, args.peak_delay, args.epsilon)
delays = np.arange(-10, 11)
correlations = np.zeros(delays.shape)
std_errors = np.zeros(delays.shape)
for i, delay in enumerate(delays):
    correlations[i], std_errors[i] = getCorrelationWithDelay(y, x, delay)
plt.plot(delays, correlations)
plt.fill_between(delays, correlations - std_errors, correlations + std_errors, color='blue', alpha=0.3, label='standard error')
plt.ylabel('Correlation Coefficient (a.u.)', fontsize='large')
plt.xlabel('Delay (time bins)', fontsize='large'); plt.xlim([-10,10]);
plt.xticks(delays); plt.legend(fontsize='large')
plt.show(block=False)
