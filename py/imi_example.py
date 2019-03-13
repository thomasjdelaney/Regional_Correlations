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

parser = argparse.ArgumentParser(description='Reproducing some results from the IMI paper.')
parser.add_argument('-v', '--underlying_variance', help='The variance of the underlying normal distributions.', type=float, default=1.0)
parser.add_argument('-e', '--epsilon', help='The strength of the connection.', type=float, default=0.5)
parser.add_argument('-n', '--num_samples', help='The number of samples to make.', type=int, default=1000)
parser.add_argument('-p', '--peak_delay', help='The delay for the effect of y on x.', type=int, default=4)
parser.add_argument('-b', '--num_bins', help='The number of bins to use in each sample', type=int, default=34)
args = parser.parse_args()

epsilon = 0.5 # strength of connection
num_samples = 1000
peak_delay = 4
num_bins = 34

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
    return a_y[:,:num_bins-peak_delay], a_x[:,peak_delay:]

def offsetSamples(y, x, offset): # positive offset indicates x should be 'ahead' relative to y
    num_bins = y.shape[1]
    if offset > num_bins: sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + ' Offset too big.')
    if offset > 0:
        offset_y = y[:, :num_bins-offset]
        offset_x = x[:, offset:]
    elif offset < 0:
        offset_y = y[:, np.abs(offset):]
        offset_x = x[:, :num_bins+offset]
    else:
        offset_y = y
        offset_x = x
    return offset_y, offset_x

def measureCorrelationWithOffset(y, x, offset):
    num_samples = y.shape[0]
    offset_x = np.roll(x, offset, axis=1)
    corr_coeffs = np.zeros(num_samples)
    p_values = corr_coeffs
    for i, (y_trial, x_trial) in enumerate(zip(y, offset_x)):
        corr_coeffs[i], p_values[i] = pearsonr(y_trial, x_trial)
    return corr_coeffs

y, x = getCorrelatedSamples(args.num_samples, args.num_bins, args.underlying_variance, args.peak_delay, args.epsilon)
offsets = np.arange(-10, 10)
correlations = np.vstack([measureCorrelationWithOffset(y, x, offset) for offset in offsets])
plt.plot(offsets, correlations.mean(axis=1))
plt.ylabel('Correlation Coefficient (a.u.)', fontsize='large')
plt.xlabel('Delay (time bins)', fontsize='large')
plt.show(block=False)
