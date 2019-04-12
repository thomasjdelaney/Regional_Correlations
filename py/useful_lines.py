import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='For plotting spike count correlations against geometric means of the product of the firing rates of the paired neurons.')
parser.add_argument('-f', '--paired_file', help='The file in which to find the correlations and mutual info.', type=str, default='all_pairs.csv')
parser.add_argument('-g', '--single_file', help='The file in which to find the firing rates.', type=str, default='all_firing_rates.csv')
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

def getDataFrames(single_file, paired_file, bin_width):
    firing_frame = pd.read_csv(os.path.join(csv_dir, args.single_file))
    pairwise_frame = pd.read_csv(os.path.join(csv_dir, args.paired_file))
    firing_frame = firing_frame[firing_frame.bin_width == bin_width]
    pairwise_frame = pairwise_frame[pairwise_frame.bin_width == bin_width]
    return firing_frame, pairwise_frame

def getWorkingCells(firing_region_frame, pairwise_region_frame):
    required_cells = np.unique(pairwise_region_frame.loc[:, ['first_cell_id', 'second_cell_id']])
    available_cells = np.unique(firing_region_frame.cell_id)
    return np.intersect1d(required_cells, available_cells)

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading single and pairwise measurements...')
    firing_frame, pairwise_frame = getDataFrames(args.single_file, args.paired_file, args.bin_width)
    for region in rc.regions:
        firing_region_frame = firing_frame[firing_frame.region == region]
        pairwise_region_frame = pairwise_frame[pairwise_frame.region == region]
        working_cells = getWorkingCells(firing_region_frame, pairwise_region_frame)


if not(args.debug):
    main()
