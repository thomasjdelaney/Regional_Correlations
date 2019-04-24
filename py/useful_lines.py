import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log

parser = argparse.ArgumentParser(description='For creating histograms of correlation coefficients.')
parser.add_argument('-f', '--filename', help='The file in which to find the correlations.', type=str, default='all_regions_stims_pairs_widths.csv')
parser.add_argument('-b', '--bin_width', help='The bin width to use for correlations.', type=float, default=1.0)
parser.add_argument('-p', '--prefix', help='A prefix for the image file names.', type=str, default='')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows',30) # setting display options for terminal display
pd.options.mode.chained_assignment = None  # default='warn'

# defining useful directories
proj_dir = os.path.join(os.environ['HOME'], 'Regional_Correlations')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')

# loading useful functions
sys.path.append(py_dir)
import regionalCorrelations as rc

def calcEntropy(x, base=2):
    if x.size <= 1:
        return 0
    values, counts = np.unique(x, return_counts=True)
    probs = counts / x.size
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    return np.array([-i*log(i,base)for i in probs]).sum()

num_vals_list = np.arange(1,101)
entropies = np.zeros(num_vals_list.shape)
firing_rates = np.zeros(num_vals_list.shape)
for i, num_vals in enumerate(num_vals_list):
    x = np.random.randint(0, num_vals, 1000)
    entropies[i] = calcEntropy(x)
    firing_rates[i] = x.sum()/1000.0

fig = plt.figure(figsize=(4,3))
plt.plot(num_vals_list-1, entropies, label='Max H(X)')
plt.plot(num_vals_list-1, np.log2(num_vals_list), label=r'$\log_2 \left( n_{\max} + 1 \right)$', color='orange')
plt.xlabel(r'Maximum observed spikes, $n_{\max}$', fontsize='large')
plt.ylabel(r'$H(X)$ (bits)', fontsize='large')
plt.xlim([0,100])
plt.ylim([0,entropies.max()])
plt.legend(fontsize='large')
plt.tight_layout()
plt.show()
