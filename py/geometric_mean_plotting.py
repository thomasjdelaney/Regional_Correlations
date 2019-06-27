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
parser.add_argument('-t', '--plot_type', help='The type of plot we want to make', type=str, default='firing', choices=['firing', 'pairwise'])
parser.add_argument('-b', '--bin_width', help='The bin width to use for correlations.', type=float, default=1.0)
parser.add_argument('-p', '--prefix', help='A prefix for the image file names.', type=str, default='')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

# defining useful directories
proj_dir = os.path.join(os.environ['PROJ'], 'Regional_Correlations')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
image_dir = os.path.join(proj_dir, 'images')

# loading useful functions
sys.path.append(py_dir)
import regionalCorrelations as rc
import regionalCorrelationsPlotting as rcp

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

def plotMeasureVsGeomMeanForRegion(working_frame, region, measure, y_lim, y_label='', is_tight=True, figsize=(4,3)):
    correlation = working_frame[['geometric_mean', measure]].corr().values[0,1]
    geom_max = working_frame.geometric_mean.max()
    text_y = -0.95 if measure=='corr_coef' else 0.05
    fig = plt.figure(figsize=figsize) if figsize != None else 0
    plt.scatter(working_frame.geometric_mean, working_frame[measure], marker='.', color=rcp.region_to_colour[region], label=region.replace('_', ' ').capitalize())
    if y_label != '':
        plt.ylabel(y_label, fontsize='x-large')
        plt.yticks(fontsize='x-large')
    else:
        plt.yticks([])
    plt.ylim(y_lim)
    plt.xlabel(r'Firing Rate Geom. Mean (Hz)', fontsize='x-large')
    plt.xticks(fontsize='x-large')
    plt.xlim([0, geom_max])
    plt.legend(fontsize='x-large', loc='upper right')
    plt.text(geom_max*0.6, text_y, r'$\rho=$' + str(round(correlation,2)), fontsize='x-large')
    plt.tight_layout() if is_tight else 0

def saveAndClose(filename, directory):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving '+ filename + '...')
    plt.savefig(os.path.join(image_dir, directory, filename))
    plt.close()

def plotMeasuresVsGeomMean(working_frame, region, stim_id, prefix):
    plotMeasureVsGeomMeanForRegion(working_frame, region, 'corr_coef', [-1,1], y_label=r'$r_{SC}$')
    filename = prefix + 'corr_vs_geometric_' + region + '_' + str(stim_id) + '.png'
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving '+ filename + '...')
    plt.savefig(os.path.join(image_dir, 'geometric_mean', filename))
    plt.close()
    plotMeasureVsGeomMeanForRegion(working_frame, region, 'mutual_info_qe', [0, working_frame.mutual_info_qe.max()], y_label=r'$I(X;Y)$ (bits)')
    filename = prefix + 'info_vs_geometric_' + region + '_' + str(stim_id) + '.png'
    saveAndClose(filename, 'geometric_mean')

def plotInfoVsCorr(pairwise_region_frame, region, max_mi, stim_id, prefix, use_ylabel=True, is_tight=True, is_save=True, figsize=(4,3)):
    plt.figure(figsize=figsize) if figsize != None else 0
    plt.scatter(pairwise_region_frame.corr_coef, pairwise_region_frame.mutual_info_qe, color=rcp.region_to_colour[region], label=region.replace('_', ' ').capitalize())
    plt.xlim([-1,1]); plt.ylim([0, max_mi]);
    z = np.polyfit(pairwise_region_frame.corr_coef, pairwise_region_frame.mutual_info_qe, 2)
    x = np.arange(-1,1.05,0.05)
    p = np.poly1d(z)
    plt.plot(x, p(x), color='black', label='Quadratic fit')
    plt.xlabel(r'$r_{SC}$', fontsize='x-large')
    plt.xticks(fontsize='x-large')
    if use_ylabel:
        plt.ylabel(r'$I(X;Y)$ (bits)', fontsize='x-large')
        plt.yticks(fontsize='x-large')
    else:
        plt.ylabel('')
        plt.yticks([])
    plt.legend(fontsize='x-large')
    plt.tight_layout() if is_tight else 0
    filename = prefix + 'info_vs_corr_' + region + '_' + str(stim_id) + '.png'
    saveAndClose(filename, 'mutual_info_vs_corr') if is_save else 0

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading single and pairwise measurements...')
    firing_frame, pairwise_frame = getDataFrames(args.single_file, args.paired_file, args.bin_width)
    if args.plot_type == 'firing':
        for region in rc.regions:
            best_stim = rc.getBestStimFromRegion(firing_frame, region)
            firing_region_frame = firing_frame[(firing_frame.region == region) & (firing_frame.stim_id == best_stim)]
            pairwise_region_frame = pairwise_frame[(pairwise_frame.region == region) & (pairwise_frame.stim_id == best_stim)]
            working_frame = getWorkingFrame(firing_region_frame, pairwise_region_frame)
            working_frame.loc[:,'geometric_mean'] = np.sqrt(working_frame.first_firing_rate * working_frame.second_firing_rate)
            plotMeasuresVsGeomMean(working_frame, region, best_stim, args.prefix)
    elif args.plot_type == 'pairwise':
        max_mi = pairwise_frame.mutual_info_qe.max()
        for region in rc.regions:
            best_stim = rc.getBestStimFromRegion(pairwise_frame, region)
            pairwise_region_frame = pairwise_frame[(pairwise_frame.region == region) & (pairwise_frame.stim_id == best_stim)]
            plotInfoVsCorr(pairwise_region_frame, region, max_mi, best_stim, args.prefix)
    else:
        sys.exit(dt.datetime.now().isoformat() + 'ERROR: ' + 'Unrecognised measure type!')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
