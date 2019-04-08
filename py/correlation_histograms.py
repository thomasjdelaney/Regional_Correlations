"""
For creating histograms of correlation coefficients.
"""
import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='For creating histograms of correlation coefficients.')
parser.add_argument('-f', '--filename', help='The file in which to find the correlations.', type=str, default='all_regions_stims_pairs_widths.csv')
parser.add_argument('-b', '--bin_width', help='The bin width to use for correlations.', type=float, default=1.0)
parser.add_argument('-p', '--prefix', help='A prefix for the image file names.', type=str, default='')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

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

def plotRegionalHistogram(correlation_frame, region, bin_width, stim_id, x_col, x_label, x_lim):
    region_bin_frame = correlation_frame[(correlation_frame.region == region)&(correlation_frame.bin_width == bin_width)&(correlation_frame.stim_id == stim_id)]
    region_bin_frame[x_col].hist(grid=False, bins=25, color=rc.region_to_colour[region], range=x_lim, label=region.replace('_', ' ').capitalize(), figsize=(4,3))
    plt.xlim(x_lim)
    plt.xlabel(x_label, fontsize='large')
    plt.ylabel('Number of pairs', fontsize='large')
    plt.title('Total pairs = ' + str(region_bin_frame.shape[0]) + ', width = ' + str(bin_width))
    plt.legend()
    plt.tight_layout()

def plotRegionalCorrelationHistogram(correlation_frame, region, stim_id, bin_width, prefix):
    plotRegionalHistogram(correlation_frame, region, bin_width, stim_id, 'corr_coef', r'$r_{SC}$', [-1,1])
    filename = prefix + region + '_' + str(stim_id) + '_' + str(bin_width).replace('.','p') + '_correlation_histogram.png'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving '+ filename + '...')
    plt.savefig(os.path.join(image_dir, 'correlation_histograms', filename))
    plt.close()

def plotRegionalInfoHistogram(correlation_frame, region, stim_id, bin_width, prefix, max_mi):
    plotRegionalHistogram(correlation_frame, region, bin_width, stim_id, 'mutual_info', r'$I(X;Y)$ (bits)', [0, max_mi])
    filename = prefix + region + '_' + str(stim_id) + '_' + str(bin_width).replace('.','p') + '_information_histogram.png'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving '+ filename + '...')
    plt.savefig(os.path.join(image_dir, 'information_histograms', filename))
    plt.close()

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
    stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
    correlation_frame = pd.read_csv(os.path.join(csv_dir, args.filename))
    max_mi = correlation_frame.mutual_info.max()
    for region in rc.regions:
        best_stim = rc.getBestStimFromRegion(correlation_frame, region)
        plotRegionalCorrelationHistogram(correlation_frame, region, best_stim, args.bin_width, args.prefix)
        plotRegionalInfoHistogram(correlation_frame, region, best_stim, args.bin_width, args.prefix, max_mi)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
