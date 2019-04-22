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
parser.add_argument('-m', '--measure_type', help='Pairwise or single cell measurement.', default='pairwise', choices=['pairwise', 'single', 'correction'], type=str)
parser.add_argument('-f', '--filename', help='The file in which to find the measurements.', type=str, default='all_pairs.csv')
parser.add_argument('-s', '--strong_filename', help='The file for strong measurements.', type=str, default='strong_pairs.csv')
parser.add_argument('-b', '--bin_width', help='The bin width to use for correlations.', type=float, default=1.0)
parser.add_argument('-p', '--prefix', help='A prefix for the image file names.', type=str, default='')
parser.add_argument('-t', '--use_title', help='Flag for using titles on the figures.', default=False, action='store_true')
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

pd.set_option('max_rows',30) # setting display options for terminal display
pd.options.mode.chained_assignment = None  # default='warn'

# loading useful functions
sys.path.append(py_dir)
import regionalCorrelations as rc

def plotRegionalHistogram(region_stim_bin_frame, region, x_col, x_label, x_lim, y_lim, use_title=False):
    region_stim_bin_frame[x_col].hist(grid=False, bins=25, color=rc.region_to_colour[region], range=x_lim, label=region.replace('_', ' ').capitalize(), figsize=(4,3))
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel(x_label, fontsize='large')
    plt.ylabel('Number of pairs', fontsize='large')
    plt.title('Number measurements = ' + str(region_stim_bin_frame.shape[0]), fontsize='large') if use_title else 0
    plt.legend()
    plt.tight_layout()

def saveAndClose(filename, directory):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving '+ filename + '...')
    plt.savefig(os.path.join(image_dir, directory, filename))
    plt.close()

def plotRegionalFiringRateHistogram(firing_frame, region, stim_id, bin_width, prefix, max_fr, use_title):
    plotRegionalHistogram(firing_frame, region, bin_width, stim_id, 'firing_rate_mean', 'Firing Rate (Hz)', [0, max_fr], [0,200], use_title=use_title)
    filename = prefix + region + '_' + str(stim_id) + '_' + str(bin_width).replace('.','p') + '_firing_rate_histogram.png'
    saveAndClose(filename, 'firing_rate_histograms')

def plotMutualInfoCorrection(region, region_stim_bin_frame, max_max):
    plt.scatter(region_stim_bin_frame.mutual_info_plugin, region_stim_bin_frame.mutual_info_qe, color='blue', label=region.replace('_', ' ').capitalize(), marker='.')
    plt.plot([0, max_max], [0, max_max], color='black')
    plt.xlabel(r'$I(X;Y)$ (bits) (plugin)', fontsize='large')
    plt.ylabel(r'$I(X;Y)$ (bits) (bias corrected)', fontsize='large')
    plt.xlim([0, max_max]); plt.ylim([0, max_max])
    plt.legend(fontsize='large')
    plt.tight_layout()

def plotSymmUncCorrection(region, region_stim_bin_frame):
    plt.scatter(region_stim_bin_frame.symm_unc_plugin, region_stim_bin_frame.symm_unc_qe, color='blue', label=region.replace('_', ' ').capitalize(), marker='.')
    plt.plot([0, 1], [0, 1], color='black')
    plt.xlabel(r'$U(X;Y)$ (a.u.) (plugin)', fontsize='large')
    plt.ylabel(r'$U(X;Y)$ (a.u.) (bias corrected)', fontsize='large')
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.tight_layout()

def plotCorrections(correlation_frame, region, stim_id, bin_width, prefix):
    region_stim_bin_frame = correlation_frame[(correlation_frame.region == region) & (correlation_frame.stim_id == stim_id) & (correlation_frame.bin_width == bin_width)]
    max_max = region_stim_bin_frame[['mutual_info_plugin', 'mutual_info_qe']].max().max()
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plotMutualInfoCorrection(region, region_stim_bin_frame, max_max)
    plt.subplot(1,2,2)
    plotSymmUncCorrection(region, region_stim_bin_frame)
    filename = prefix + region + '_' + str(stim_id) + '_' + str(bin_width).replace('.', 'p') + '_corrections.png'
    saveAndClose(filename, 'corrections')

def getCorrFrame(filename, bin_width):
    correlation_frame = pd.read_csv(os.path.join(csv_dir, filename))
    return correlation_frame[correlation_frame.bin_width == bin_width]

def getBestStimsLims(corr_bin_frame, strong_corr_bin_frame, measure, hist_range=None):
    hist_range = [0, corr_bin_frame[measure].max()] if hist_range == None else hist_range
    best_stims = np.zeros(len(rc.regions), dtype=int)
    top_count = 0
    max_bin = 0
    for i, region in enumerate(rc.regions):
        best_stims[i] = rc.getBestStimFromRegion(strong_corr_bin_frame, region)
        corr_region_bin_stim = corr_bin_frame[(corr_bin_frame.stim_id == best_stims[i]) & (corr_bin_frame.region == region)]
        counts, bins = np.histogram(corr_region_bin_stim[measure], bins=25, range=hist_range)
        top_count = np.max([top_count, counts.max()])
        max_bin = np.max([max_bin, bins[-1]])
    return best_stims, [0, 10*np.ceil(top_count/10).astype(int)], max_bin

def plotRegionalCorrelationHistograms(corr_bin_frame, corr_best_stims, corr_y_lim, use_title, prefix, bin_width):
    for region, best_stim in zip(rc.regions, corr_best_stims):
        region_stim_bin_frame = corr_bin_frame[(corr_bin_frame.region == region) & (corr_bin_frame.stim_id == best_stim)]
        plotRegionalHistogram(region_stim_bin_frame, region, 'corr_coef', r'$r_{SC}$', [-1,1], corr_y_lim, use_title=use_title)
        filename = prefix + region + '_' + str(best_stim) + '_' + str(bin_width).replace('.','p') + '_correlation_histogram.png'
        saveAndClose(filename, 'correlation_histograms')
    return None

def plotRegionalInformationHistograms(corr_bin_frame, info_best_stims, info_y_lim, info_max_bin, use_title, prefix, bin_width):
    for region, best_stim in zip(rc.regions, info_best_stims):
        region_stim_bin_frame = corr_bin_frame[(corr_bin_frame.region == region) & (corr_bin_frame.stim_id == best_stim)]
        plotRegionalHistogram(region_stim_bin_frame, region, 'mutual_info_qe', r'$I(X;Y)$ (bits)', [0, info_max_bin], info_y_lim, use_title=use_title)
        filename = prefix + region + '_' + str(best_stim) + '_' + str(bin_width).replace('.','p') + '_information_histogram.png'
        saveAndClose(filename, 'information_histograms')
    return None

def plotRegionalSymmUncHistograms(corr_bin_frame, symm_best_stims, symm_y_lim, use_title, prefix, bin_width):
    for region, best_stim in zip(rc.regions, symm_best_stims):
        region_stim_bin_frame = corr_bin_frame[(corr_bin_frame.region == region) & (corr_bin_frame.stim_id == best_stim)]
        plotRegionalHistogram(region_stim_bin_frame, region, 'symm_unc_qe', r'$U(X;Y)$ (a.u.)', [0, 1], symm_y_lim, use_title=use_title)
        filename = prefix + region + '_' + str(best_stim) + '_' + str(bin_width).replace('.', 'p') + '_symm_unc_histogram.png'
        saveAndClose(filename, 'symm_unc_histograms')
    return None

def plotRegionalFiringRateHistograms(firing_bin_frame, firing_best_stims, firing_y_lim, firing_max_bin, use_title, prefix, bin_width):
    for region, best_stim in zip(rc.regions, firing_best_stims):
        region_stim_bin_frame = firing_bin_frame[(firing_bin_frame.region == region) & (firing_bin_frame.stim_id == best_stim)]
        plotRegionalHistogram(region_stim_bin_frame, region, 'firing_rate_mean', r'Firing rate (Hz)', [0, firing_max_bin], firing_y_lim, use_title=use_title)
        filename = prefix + region + '_' + str(best_stim) + '_' + str(bin_width).replace('.', 'p') + '_firing_rate_histogram.png'
        saveAndClose(filename, 'firing_rate_histograms')
    return None

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading data from disc...')
    corr_bin_frame = getCorrFrame(args.filename, args.bin_width)
    strong_corr_bin_frame = getCorrFrame(args.strong_filename, args.bin_width)
    if args.measure_type == 'pairwise':
        corr_best_stims, corr_y_lim, corr_max_bin = getBestStimsLims(corr_bin_frame, strong_corr_bin_frame, 'corr_coef', [-1, 1])
        plotRegionalCorrelationHistograms(corr_bin_frame, corr_best_stims, corr_y_lim, args.use_title, args.prefix, args.bin_width)
        info_best_stims, info_y_lim, info_max_bin = getBestStimsLims(corr_bin_frame, strong_corr_bin_frame, 'mutual_info_qe')
        plotRegionalInformationHistograms(corr_bin_frame, info_best_stims, info_y_lim, info_max_bin, args.use_title, args.prefix, args.bin_width)
        symm_best_stims, symm_y_lim, symm_max_bin = getBestStimsLims(corr_bin_frame,strong_corr_bin_frame, 'symm_unc_qe', [0, 1])
        plotRegionalSymmUncHistograms(corr_bin_frame, symm_best_stims, symm_y_lim, args.use_title, args.prefix, args.bin_width)
    elif args.measure_type == 'single':
        firing_best_stims, firing_y_lim, firing_max_bin = getBestStimsLims(corr_bin_frame, strong_corr_bin_frame, 'firing_rate_mean') 
        plotRegionalFiringRateHistograms(corr_bin_frame, firing_best_stims, firing_y_lim, firing_max_bin, args.use_title, args.prefix, args.bin_width)
    elif args.measure_type == 'correction':
        for region in rc.regions:
            best_stim = rc.getBestStimFromRegion(correlation_frame, region)
            plotCorrections(correlation_frame, region, best_stim, args.bin_width, args.prefix)
    else:
        sys.exit(dt.datetime.now().isoformat() + 'ERROR: ' + 'Unrecognised measure type!')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
