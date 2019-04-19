import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

all_pairs = pd.read_csv(os.path.join(csv_dir, args.filename))
for region in rc.regions:
    region_bin_frame = all_pairs[(all_pairs.region == region) & (all_pairs.bin_width == args.bin_width)]
    best_stim = rc.getBestStimFromRegion(correlation_frame, region)
    region_stim_bin_frame = region_bin_frame[region_bin_frame.stim_id == best_stim]
    fig = plt.figure()
    plt.subplot(1,2,1)
    plotMutualInfoCorrection(region, region_stim_bin_frame, region_stim_bin_frame[['mutual_info_plugin', 'mutual_info_qe']].max().max())
    plt.subplot(1,2,2)
    plotSymmUncCorrection(region, region_stim_bin_frame)
    plt.show(block=False)
