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
from scipy.stats import entropy
from pyitlib import discrete_random_variable as drv

parser = argparse.ArgumentParser(description='Reproducing some results from the IMI paper.')
parser.add_argument('-v', '--underlying_variance', help='The variance of the underlying normal distributions.', type=float, default=1.0)
parser.add_argument('-e', '--epsilon', help='The strength of the connection.', type=float, default=0.5)
parser.add_argument('-n', '--num_samples', help='The number of samples to make.', type=int, default=1000)
parser.add_argument('-p', '--peak_delay', help='The delay for the effect of y on x.', type=int, default=4)
parser.add_argument('-b', '--num_bins', help='The number of bins to use in each sample', type=int, default=30)
parser.add_argument('-w', '--filter_window', help='The number of time bins that the filter will effect on either side of the peak.', type=int, default=3)
parser.add_argument('-g', '--filter_variance', help='The variance of the gaussian filter applied to y.', type=float, default=1.0)
parser.add_argument('-f', '--num_future_past_bins', help='The number of bins to use as the past and future.', type=int, default=2)
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

def getCorrelatedSamples(num_samples, num_bins, underlying_variance, peak_delay, epsilon, filter_var=1, filter_window=3):
    s_x = normal(0,underlying_variance,[num_samples,num_bins])
    s_y = normal(0,underlying_variance,[num_samples,num_bins])
    a_y = np.min([np.ones(s_y.shape), np.max([np.zeros(s_y.shape),np.floor(s_y)], axis=0)], axis=0)
    filtered_y = getFilteredSpikeTrains(a_y, filter_var, filter_window)
    under_x = s_x + epsilon*filtered_y
    a_x = np.min([np.ones(s_x.shape), np.max([np.zeros(s_x.shape), np.floor(under_x)], axis=0)], axis=0)
    return a_y, np.roll(a_x, peak_delay, axis=1)

def getFuturePastIndsFromIndex(index, future_past):
    return np.concatenate([index + np.arange(-future_past,0), index + np.arange(1, future_past+1)])

def getIndexPairsWithDelay(num_samples, num_bins, delay, num_future_past_bins):
    index_pairs = np.array([])
    y_future_past_inds = np.array([])
    x_future_past_inds = np.array([])
    for i,j in product(np.arange(num_future_past_bins, num_bins-num_future_past_bins), np.arange(num_future_past_bins, num_bins-num_future_past_bins)):
        if i-j == -delay:
            index_pairs = np.append(index_pairs,[i,j])
            y_future_past_inds = np.append(y_future_past_inds, getFuturePastIndsFromIndex(i, num_future_past_bins))
            x_future_past_inds = np.append(x_future_past_inds, getFuturePastIndsFromIndex(j, num_future_past_bins))
    num_pairs = index_pairs.size//2
    index_pairs = np.reshape(index_pairs, [num_pairs, 2]).astype(int)
    y_future_past_inds = np.reshape(y_future_past_inds, [num_pairs, 2*num_future_past_bins]).astype(int)
    x_future_past_inds = np.reshape(x_future_past_inds, y_future_past_inds.shape).astype(int)
    return num_pairs, index_pairs, y_future_past_inds, x_future_past_inds

def getCorrelationWithDelay(y, x, delay):
    num_samples, num_bins = y.shape
    num_pairs, index_pairs, yfpi, xfpi = getIndexPairsWithDelay(num_samples, num_bins, delay, 0)
    correlations = np.zeros(num_pairs)
    p_values = np.zeros(num_pairs)
    for ind, (i, j) in enumerate(index_pairs):
        correlations[ind], p_values[ind] = pearsonr(y[:,i], x[:,j])
    return correlations.mean(), correlations.std()/np.sqrt(num_pairs)

def calcEntropy(samples):
    num_samples = samples.shape[0]
    labels, counts = np.unique(samples, return_counts=True, axis=0)
    if labels.size <= 1:
        return 0
    probs = counts/num_samples
    return -(probs * np.log2(probs)).sum()

def calcIncrementalMutualInformation(x_samples, y_samples, future_past):
    future_past_entropy = calcEntropy(future_past)
    x_future_past = np.hstack([x_samples.reshape([x_samples.size,1]), future_past])
    y_future_past = np.hstack([y_samples.reshape([y_samples.size,1]), future_past])
    x_y_future_past = np.hstack([y_samples.reshape([y_samples.size,1]), x_future_past])
    x_future_past_joint_entropy = calcEntropy(x_future_past)
    y_future_past_joint_entropy = calcEntropy(y_future_past)
    x_y_future_past_joint_entropy = calcEntropy(x_y_future_past)
    x_given_future_past_cond_entropy = x_future_past_joint_entropy - future_past_entropy
    x_given_y_future_past_cond_entropy = x_y_future_past_joint_entropy - y_future_past_joint_entropy
    return (x_given_future_past_cond_entropy - x_given_y_future_past_cond_entropy)/x_given_future_past_cond_entropy

def getIMIWithDelay(y, x, delay, num_future_past_bins):
    num_samples, num_bins = y.shape
    num_pairs, index_pairs, y_future_past_inds, x_future_past_inds = getIndexPairsWithDelay(num_samples, num_bins, delay, num_future_past_bins)
    imi = np.zeros(num_pairs)
    for ind, (index_pair, xfpi, yfpi) in enumerate(zip(index_pairs, x_future_past_inds, y_future_past_inds)):
        y_ind, x_ind = index_pair
        x_samples = x[:,x_ind]; y_samples = y[:,y_ind];
        x_future_past = x[:, xfpi]; y_future_past = y[:, yfpi];
        future_past = np.hstack([x_future_past, y_future_past])
        imi[ind] = calcIncrementalMutualInformation(x_samples, y_samples, future_past)
    return imi.mean(), imi.std()/np.sqrt(num_pairs)

def plotWithStdErrors(measure, std_errors, delays, ylabel):
    plt.plot(delays, measure)
    plt.fill_between(delays, measure - std_errors, measure + std_errors, color='blue', alpha=0.3, label='standard error')
    plt.ylabel(ylabel, fontsize='large')
    plt.xlabel('Delay (time bins)', fontsize='large')
    plt.xlim([delays.min(), delays.max()])
    plt.xticks(delays)
    plt.legend(fontsize='large')

y, x = getCorrelatedSamples(args.num_samples, args.num_bins, args.underlying_variance, args.peak_delay, args.epsilon, filter_var=args.filter_variance, filter_window=args.filter_window)
delays = np.arange(-10, 11)
correlations = np.zeros(delays.shape)
corr_std_errors = np.zeros(delays.shape)
imis = np.zeros(delays.shape)
imi_std_errors = np.zeros(delays.shape)
for i, delay in enumerate(delays):
    correlations[i], corr_std_errors[i] = getCorrelationWithDelay(y, x, delay)
    imis[i], imi_std_errors[i] = getIMIWithDelay(y, x, delay, args.num_future_past_bins)
plt.subplot(2,1,1)
plotWithStdErrors(correlations, corr_std_errors, delays, "Correlation Coefficient (a.u.)")
plt.subplot(2,1,2)
plotWithStdErrors(imis, imi_std_errors, delays, "Incremental Mutual Information (a.u.)")
plt.show(block=False)
