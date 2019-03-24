"""
For plotting the data found in 'all_regions_stims_pairs_widths.csv'
"""
import os, sys, argparse
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="For plotting the data found in 'all_regions_stims_pairs_widths.csv'.")
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows',30) # setting display options for terminal display
pd.options.mode.chained_assignment = None  # default='warn'

# defining useful directories
proj_dir = os.path.join(os.environ['HOME'], 'Regional_Correlations')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')

def getBestStimFromRegion(all_regions_stims_pairs_widths, region):
    region_frame = all_regions_stims_pairs_widths[all_regions_stims_pairs_widths.region == region]
    return region_frame['stim_id'].value_counts().index[0]

def plotCorrCoefByBinWidthForRegionStim(all_regions_stims_pairs_widths, region, stim_id, region_to_colour):
    region_stim_frame = all_regions_stims_pairs_widths[(all_regions_stims_pairs_widths.region == region)&(all_regions_stims_pairs_widths.stim_id == stim_id)]
    region_stim_frame.loc[:,'corr_coef'] = region_stim_frame.loc[:,'corr_coef'].abs()
    agg_frame = region_stim_frame[['bin_width','corr_coef']].groupby('bin_width').agg({'corr_coef':['mean', 'std']})
    agg_frame.loc[:,'std_err'] = agg_frame.corr_coef.loc[:,'std']/np.sqrt(agg_frame.shape[0])
    fig = plt.figure(figsize=(4,3))
    plt.plot(agg_frame.corr_coef.index.values, agg_frame.corr_coef['mean'], color=region_to_colour[region], label=region.replace('_', ' '))
    plt.fill_between(agg_frame.corr_coef.index.values, agg_frame.corr_coef['mean']-agg_frame['std_err'], agg_frame.corr_coef['mean']+agg_frame['std_err'], color=region_to_colour[region], alpha=0.3)
    plt.xlim([agg_frame.corr_coef.index.min(), agg_frame.corr_coef.index.max()])
    plt.xscale('log')
    plt.ylim([0,1])
    plt.xlabel('Bin width (s)', fontsize='large')
    plt.ylabel('|Corr Coef| (a.u.)', fontsize='large')
    plt.legend(fontsize='large')
    plt.tight_layout()
    filename = 'bin_width_correlations_' + region + '_' + str(stim_id) + '.png'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving '+filename+'...')
    plt.savefig(os.path.join(image_dir, 'correlations_vs_bin_width', filename))
    plt.close()

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading all_regions_stims_pairs_widths.csv...')
    all_regions_stims_pairs_widths = pd.read_csv(os.path.join(csv_dir, 'all_regions_stims_pairs_widths.csv'))
    regions = ['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1']
    colours = cm.gist_rainbow(np.linspace(0, 1, 5)) # 5 regions
    region_to_colour = dict(list(zip(regions, colours)))
    for i,region in enumerate(regions):
        best_stim = getBestStimFromRegion(all_regions_stims_pairs_widths, region)
        plotCorrCoefByBinWidthForRegionStim(all_regions_stims_pairs_widths, region, best_stim, region_to_colour)

if not(args.debug):
    main()
