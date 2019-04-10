import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Calculate pairwise correlations between given choice of neurons.')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
parser.add_argument('-s', '--numpy_seed', help='The seed to use to initialise numpy.random.', default=1798, type=int)
args = parser.parse_args()

np.random.seed(args.numpy_seed) # setting seed
pd.set_option('max_rows',30) # setting display options for terminal display

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

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
cell_info, id_adjustor = rc.loadCellInfo(csv_dir)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading stim info...')
stim_info = loadmat(os.path.join(mat_dir, 'experiment2stimInfo.mat'))
region = rc.regions[0]
stim_id = 2
group = 'good'
is_strong = True
bin_width = 2.0
cell_ids = cell_info[(cell_info.region==region)&(cell_info.group==group)].index.values
trials_info = rc.getStimTimesIds(stim_info, stim_id)
spike_time_dict = rc.loadSpikeTimes(posterior_dir, frontal_dir, cell_ids, id_adjustor)
responding_cells = rc.getRespondingCells(cell_ids, trials_info, spike_time_dict, cell_info, num_cells=0, is_strong=True, strong_threshold=20.0)
