"""
For plotting the data found in 'all_regions_stims_pairs_widths.csv'
"""
import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="For plotting the data found in 'all_regions_stims_pairs_widths.csv'.")
parser.add_argument('-c', '--correlation_type', help='Signal correlation or spike count correlation.', default='spike_count', choices=['spike_count', 'signal', 'bifurcation', 'mutual_info', 'firing_rate'], type=str)
parser.add_argument('-f', '--filename', help='The file from which to get the pairwise correlation data.', type=str, default='all_regions_stims_pairs_widths.csv')
parser.add_argument('-p', '--image_file_prefix', help='A prefix for the image file names.', type=str, default='')
parser.add_argument('-x', '--x_axis_scale', help='The scale of the x-axis', type=str, default='linear', choices=['log', 'linear'])
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

def plotColumnByBinWidth(region_stim_frame, y_column, region, x_axis_scale, y_label, y_lim=[0,1]):
    agg_frame = region_stim_frame[['bin_width', y_column]].groupby('bin_width').agg({y_column:['mean', 'std']})
    counts_frame = region_stim_frame['bin_width'].value_counts()
    agg_frame.loc[:, 'num_samples'] = counts_frame
    agg_frame.loc[:, 'std_err'] = agg_frame[y_column].loc[:,'std']/np.sqrt(agg_frame['num_samples'])
    fig = plt.figure(figsize=(4,3))
    plt.plot(agg_frame[y_column].index.values, agg_frame[y_column]['mean'], color=rc.region_to_colour[region], label=region.replace('_', ' ').capitalize())
    plt.fill_between(agg_frame[y_column].index.values, agg_frame[y_column]['mean']-agg_frame['std_err'], agg_frame[y_column]['mean']+agg_frame['std_err'], color=rc.region_to_colour[region], alpha=0.3)
    plt.xlim([agg_frame[y_column].index.min(), agg_frame[y_column].index.max()])
    plt.xscale('log') if x_axis_scale=='log' else 0
    plt.ylim(y_lim)
    plt.xlabel('Bin width (s)', fontsize='large')
    plt.ylabel(y_label, fontsize='large')
    plt.legend(fontsize='large')
    plt.tight_layout()

def plotAbsCorrCoefByBinWidthForRegionStim(correlation_frame, region, stim_id, prefix, x_axis_scale):
    region_stim_frame = correlation_frame[(correlation_frame.region == region)&(correlation_frame.stim_id == stim_id)]
    region_stim_frame.loc[:,'corr_coef'] = region_stim_frame.loc[:,'corr_coef'].abs()
    plotColumnByBinWidth(region_stim_frame, 'corr_coef', region, x_axis_scale, r'$|r_{SC}|$')
    filename = prefix + 'bin_width_correlations_' + region + '_' + str(stim_id) + '.png'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving ' + filename + '...')
    plt.savefig(os.path.join(image_dir, 'correlations_vs_bin_width', 'absolute_correlations', x_axis_scale, filename))
    plt.close()

def plotAbsSignalCorrByBinWidth(correlation_frame, region, prefix, x_axis_scale):
    region_frame = correlation_frame[correlation_frame.region==region]
    region_frame.columns = ['region', 'first_cell_id', 'second_cell_id', 'corr_coef', 'p_values', 'bin_width']
    region_frame.loc[:,'corr_coef'] = region_frame.loc[:,'corr_coef'].abs()
    plotColumnByBinWidth(region_frame, 'corr_coef', region, x_axis_scale, r'$|r_{signal}|$')
    filename = prefix + 'bin_width_correlations_' + region + '_0.png'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving ' + filename + '...')
    plt.savefig(os.path.join(image_dir, 'correlations_vs_bin_width', 'signal_correlations', x_axis_scale, filename))
    plt.close()

def plotFiringRateByBinWidth(firing_frame, prefix):
    y_lim = [0, firing_frame.firing_rate_mean.max()]
    for region in rc.regions:
        best_stim = rc.getBestStimFromRegion(firing_frame, region)
        region_stim_frame = firing_frame[(firing_frame.region == region) & (firing_frame.stim_id == best_stim)]
        plotColumnByBinWidth(region_stim_frame, 'firing_rate_mean', region, 'linear', 'Firing Rate (Hz)', y_lim=y_lim)
        filename = prefix + 'firing_rate_by_bin_width_' + region + '_' + str(best_stim) + '.png'
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving ' + filename + '...')
        plt.savefig(os.path.join(image_dir, 'firing_rate_vs_bin_width', filename))
        plt.close()

def plotMutualInfoByBinWidth(correlation_frame, prefix, x_axis_scale):
    y_lim = [0, correlation_frame.mutual_info.max()]
    for region in rc.regions:
        best_stim = rc.getBestStimFromRegion(correlation_frame, region)
        region_stim_frame = correlation_frame[(correlation_frame.region == region)&(correlation_frame.stim_id == best_stim)]
        plotColumnByBinWidth(region_stim_frame, 'mutual_info', region, x_axis_scale, r'$I(X;Y)$ (bits)', y_lim=y_lim)
        filename = prefix + 'mutual_info_by_bin_width_' + region + '_' + str(best_stim) + '.png'
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving ' + filename + '...')
        plt.savefig(os.path.join(image_dir, 'mutual_info_vs_bin_width', x_axis_scale, filename))
        plt.close()

def getPositiveNegativePairs(region_stim_frame):
    unique_pairs = np.unique(region_stim_frame[['first_cell_id', 'second_cell_id']].values, axis=0)
    negative_pairs = np.array([])
    positive_pairs = np.array([])
    for pair in unique_pairs:
        pair_frame = region_stim_frame.loc[(region_stim_frame.first_cell_id == pair[0])&(region_stim_frame.second_cell_id == pair[1])]
        pair_mean = pair_frame['corr_coef'].mean()
        if pair_mean < 0:
            negative_pairs = np.hstack([negative_pairs, pair])
        else:
            positive_pairs = np.hstack([positive_pairs, pair])
    num_positive_pairs = positive_pairs.size//2
    num_negative_pairs = negative_pairs.size//2
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Num positive pairs = '+ str(num_positive_pairs) + '...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Num positive pairs = '+ str(num_negative_pairs) + '...')
    return positive_pairs.reshape([num_positive_pairs, 2]).astype(int), negative_pairs.reshape([num_negative_pairs, 2]).astype(int)

def plotPositiveOrNegative(pd_frame, region, colour, x_axis_scale, frame_label):
    agg_frame = pd_frame[['bin_width','corr_coef']].groupby('bin_width').agg({'corr_coef':['mean', 'std']})
    counts_frame = region_stim_frame['bin_width'].value_counts()
    agg_frame.loc[:, 'num_samples'] = counts_frame
    agg_frame.loc[:, 'std_err'] = agg_frame.coer_coeff.loc[:,'std']/np.sqrt(agg_frame['num_samples'])
    plt.plot(agg_frame.corr_coef.index.values, agg_frame.corr_coef['mean'], color=colour, label=frame_label)
    plt.fill_between(agg_frame.corr_coef.index.values, agg_frame.corr_coef['mean']-agg_frame['std_err'], agg_frame.corr_coef['mean']+agg_frame['std_err'], color=colour, alpha=0.3)

def plotBifurcatedCorrelationsByBinWidth(correlation_frame, region, stim_id, prefix, x_axis_scale):
    region_stim_frame = correlation_frame[(correlation_frame.region == region)&(correlation_frame.stim_id == stim_id)]
    positive_pairs, negative_pairs = getPositiveNegativePairs(region_stim_frame)
    positive_frame = pd.concat([region_stim_frame.loc[(region_stim_frame.first_cell_id == p[0])&(region_stim_frame.second_cell_id == p[1])]for p in positive_pairs], ignore_index=True)
    negative_frame = pd.concat([region_stim_frame.loc[(region_stim_frame.first_cell_id == p[0])&(region_stim_frame.second_cell_id == p[1])]for p in negative_pairs], ignore_index=True)
    fig = plt.figure(figsize=(4,3))
    plotPositiveOrNegative(positive_frame, region, 'blue', x_axis_scale, 'Correlated')
    plotPositiveOrNegative(negative_frame, region, 'orange', x_axis_scale, 'Anti-correlated')
    plt.plot([0, rc.bin_widths.max()], [0,0], color='black', linestyle='dashed')
    plt.xlim([rc.bin_widths.min(), rc.bin_widths.max()])
    plt.xscale('log') if x_axis_scale=='log' else 0
    plt.ylim([-1,1])
    plt.xlabel('Bin width (s)', fontsize='large')
    plt.ylabel(r'$r_{SC}$', fontsize='large')
    plt.legend(fontsize='large', loc='lower left') if region=='hippocampus' else 0
    plt.title(region.replace('_', ' ').capitalize(), fontsize='large')
    plt.tight_layout()
    filename = prefix + 'bin_width_relative_correlations_' + region + '_' + str(stim_id) + '.png'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving '+ filename + '...')
    plt.savefig(os.path.join(image_dir, 'correlations_vs_bin_width', 'relative_correlations', x_axis_scale, filename))
    plt.close()

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading all_regions_stims_pairs_widths.csv...')
    correlation_frame = pd.read_csv(os.path.join(csv_dir, args.filename))
    if args.correlation_type == 'spike_count':
        for region in rc.regions:
            best_stim = rc.getBestStimFromRegion(correlation_frame, region)
            plotAbsCorrCoefByBinWidthForRegionStim(correlation_frame, region, best_stim, args.image_file_prefix, args.x_axis_scale)
    elif args.correlation_type == 'signal':
        for region in rc.regions:
            plotAbsSignalCorrByBinWidth(correlation_frame, region, args.image_file_prefix, args.x_axis_scale)
    elif args.correlation_type == 'bifurcation':
        for region in rc.regions:
            best_stim = rc.getBestStimFromRegion(correlation_frame, region)
            plotBifurcatedCorrelationsByBinWidth(correlation_frame, region, best_stim, args.image_file_prefix, args.x_axis_scale)
    elif args.correlation_type == 'mutual_info':
        plotMutualInfoByBinWidth(correlation_frame, args.image_file_prefix, args.x_axis_scale)
    elif args.correlation_type == 'firing_rate':
        plotFiringRateByBinWidth(correlation_frame, args.image_file_prefix)
    else:
            sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unrecognised correlation type.')

if not(args.debug):
    main()
