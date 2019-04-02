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
import matplotlib.cm as cm
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

def plotRegionalHistogram(correlation_frame, region, region_to_colour, bin_width, stim_id, prefix):
    region_bin_frame = correlation_frame[(correlation_frame.region == region)&(correlation_frame.bin_width == bin_width)&(correlation_frame.stim_id == stim_id)]
    region_bin_frame.corr_coef.hist(grid=False, bins=25, color=region_to_colour[region], range=[-1, 1], label=region.replace('_', ' ').capitalize(), figsize=(4,3))
    plt.xlim([-1, 1])
    plt.xlabel(r'$r_{SC}$', fontsize='large')
    plt.ylabel('Number of pairs', fontsize='large')
    plt.title('Total pairs = ' + str(region_bin_frame.shape[0]) + ', width = ' + str(bin_width))
    plt.legend()
    plt.tight_layout()
    filename = prefix + region + '_' + str(stim_id) + '_' + str(bin_width) + '_correlation_histogram.png'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving '+ filename + '...')
    plt.savefig(os.path.join(image_dir, 'correlation_histograms', filename))
    plt.close()

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
    stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
    correlation_frame = pd.read_csv(os.path.join(csv_dir, args.filename))
    colours = cm.gist_rainbow(np.linspace(0, 1, 5)) # 5 regions
    region_to_colour = dict(list(zip(rc.regions, colours)))
    for region in rc.regions:
        best_stim = rc.getBestStimFromRegion(correlation_frame, region)
        plotRegionalHistogram(correlation_frame, region, region_to_colour, args.bin_width, best_stim, args.prefix)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
