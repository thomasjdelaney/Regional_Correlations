"""
For carrying out statistical tests on histograms of data.
"""
import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from itertools import combinations
from scipy.io import loadmat
from scipy.stats import ks_2samp

parser = argparse.ArgumentParser(description='For carrying out statistical tests on histograms of data.')
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

def getRegionToSampleDict(correlation_frame, bin_width, measure):
    region_to_sample = {}
    for region in rc.regions:
        best_stim = rc.getBestStimFromRegion(correlation_frame, region)
        region_bin_frame = correlation_frame[(correlation_frame.region == region)&(correlation_frame.bin_width == bin_width)&(correlation_frame.stim_id == best_stim)]
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Mean ' + measure + ' = ' + str(region_bin_frame[measure].mean()))
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Median ' + measure + ' = ' + str(region_bin_frame[measure].median()))
        region_to_sample[region] = region_bin_frame.loc[:,measure]
    return region_to_sample

def getKSStatPVal(regions, region_to_corr, region_to_info):
    corr_stat, corr_p_val = ks_2samp(region_to_corr[regions[0]], region_to_corr[regions[1]])
    info_stat, info_p_val = ks_2samp(region_to_info[regions[0]], region_to_info[regions[1]])
    return regions[0], regions[1], corr_stat, corr_p_val, info_stat, info_p_val

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
    stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
    correlation_frame = pd.read_csv(os.path.join(csv_dir, args.filename))
    max_mi = correlation_frame.mutual_info.max()
    region_to_corr = getRegionToSampleDict(correlation_frame, args.bin_width, 'corr_coef')
    region_to_info = getRegionToSampleDict(correlation_frame, args.bin_width, 'mutual_info')
    region_combinations = combinations(rc.regions, 2)
    stats_p_values = np.array([getKSStatPVal(regions, region_to_corr, region_to_info) for regions in region_combinations])
    stat_test_frame = pd.DataFrame(stats_p_values, columns=['first_region', 'second_region', 'corr_stat', 'corr_p_value', 'info_stat', 'info_p_value'])
    for col in ['corr_stat', 'corr_p_value', 'info_stat', 'info_p_value']:
        stat_test_frame.loc[:,col] = stat_test_frame.loc[:,col].astype(float)
    stat_test_frame.loc[:,'is_corr_sig'] = stat_test_frame.loc[:,'corr_p_value'] < 0.05
    stat_test_frame.loc[:,'is_info_sig'] = stat_test_frame.loc[:,'info_p_value'] < 0.05
    print(stat_test_frame)
    filename = args.prefix + 'stats_tests.csv'
    stat_test_frame.to_csv(os.path.join(csv_dir, filename), index=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + filename + ' saved.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
