import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

all_pairs = pd.read_csv(os.path.join(csv_dir, 'all_pairs.csv'))
region = rc.regions[1]
stim_id = 8
bin_width = 1.0
for region in rc.regions:
    region_stim_bin_frame = all_pairs[(all_pairs.region == region) & (all_pairs.stim_id == stim_id) & (all_pairs.bin_width == bin_width)]
    max_max = region_stim_bin_frame[['mutual_info_plugin', 'mutual_info_qe', 'mutual_info_pt']].max().max()
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title(region, fontsize='large')
    plt.scatter(region_stim_bin_frame.mutual_info_plugin, region_stim_bin_frame.mutual_info_qe, color='orange', label='qe', marker='.')
    plt.plot([0, max_max], [0, max_max], color='black')
    plt.xlabel(r'$I(X;Y)$ (bits) (plugin)', fontsize='large')
    plt.ylabel(r'$I(X;Y)$ (bits) (bias corrected)', fontsize='large')
    plt.xlim([0, max_max]); plt.ylim([0, max_max])
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.subplot(1,2,2)
    plt.scatter(region_stim_bin_frame.mutual_info_plugin, region_stim_bin_frame.mutual_info_pt, color='blue', label='pt', marker='.')
    max_max = np.max([region_stim_bin_frame.mutual_info_plugin.max(), region_stim_bin_frame.mutual_info_pt.max()])
    plt.plot([0, max_max], [0, max_max], color='black')
    plt.xlabel(r'$I(X;Y)$ (bits) (plugin)', fontsize='large')
    plt.xlim([0, max_max]); plt.ylim([0, max_max])
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.show(block=False)
