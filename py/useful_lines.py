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
    firing_frame = pd.read_csv(os.path.join(csv_dir, args.single_file), usecols=lambda x: x != 'Unnamed: 0')
    pairwise_frame = pd.read_csv(os.path.join(csv_dir, args.paired_file), usecols=lambda x: x != 'Unnamed: 0')
    firing_frame = firing_frame[firing_frame.bin_width == bin_width]
    pairwise_frame = pairwise_frame[pairwise_frame.bin_width == bin_width]
    return firing_frame, pairwise_frame

def getWorkingFrame(firing_region_frame, pairwise_region_frame):
    required_cells = np.unique(pairwise_region_frame.loc[:, ['first_cell_id', 'second_cell_id']])
    available_cells = np.unique(firing_region_frame.cell_id)
    working_cells = np.intersect1d(required_cells, available_cells)
    excluded_cells = np.setdiff1d(required_cells, available_cells)
    working_pairwise = pairwise_region_frame[np.isin(pairwise_region_frame.first_cell_id, excluded_cells, invert=True)&np.isin(pairwise_region_frame.second_cell_id, excluded_cells, invert=True)]
    working_firing = firing_region_frame[np.isin(firing_region_frame.cell_id, working_cells)]
    working_firing.rename(index=str, columns={'cell_id':'first_cell_id', 'firing_rate_mean':'first_firing_rate'}, inplace=True)
    working_pairwise = working_pairwise.merge(working_firing[['first_cell_id', 'stim_id', 'first_firing_rate']], on=['first_cell_id', 'stim_id'])
    working_firing.rename(index=str, columns={'first_cell_id':'second_cell_id', 'first_firing_rate':'second_firing_rate'}, inplace=True)
    working_pairwise = working_pairwise.merge(working_firing[['second_cell_id', 'stim_id', 'second_firing_rate']], on=['second_cell_id', 'stim_id'])
    return working_pairwise

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading single and pairwise measurements...')
    firing_frame, pairwise_frame = getDataFrames(args.single_file, args.paired_file, args.bin_width)
    for region in rc.regions:
        firing_region_frame = firing_frame[firing_frame.region == region]
        pairwise_region_frame = pairwise_frame[pairwise_frame.region == region]
        working_frame = getWorkingFrame(firing_region_frame, pairwise_region_frame)
        working_frame.loc[:,'geometric_mean'] = np.sqrt(working_frame.first_firing_rate * working_frame.second_firing_rate)


if not(args.debug):
    main()
