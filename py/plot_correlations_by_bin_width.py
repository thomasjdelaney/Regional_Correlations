import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

proj_dir = os.path.join(os.environ['HOME'], 'Regional_Correlations')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images', 'correlations_vs_bin_width')

data_file = os.path.join(csv_dir, 'correlations_by_bin_length.csv')
data_frame = pd.read_csv(data_file)

first_cell_id = data_frame.first_cell_id[0]
second_cell_id = data_frame.second_cell_id[0]
stim_id = np.unique(data_frame.stim_id)[0]
region = np.unique(data_frame.region)[0]

plt.plot(data_frame.bin_length, data_frame.corr_coef)
plt.xlim([data_frame.bin_length.min(), data_frame.bin_length.max()])
plt.ylim([-1, 1])
plt.xlabel("Bin width (s)", fontsize="large")
plt.ylabel("Correlation Coefficient (a.u.)", fontsize="large")
plot_title = "Stim ID:" + str(stim_id) + ", Region:" + region + ", Cell IDs:" + str([first_cell_id, second_cell_id])
plt.title(plot_title)
file_name = region + "_" + str(stim_id) + "_" + str(first_cell_id) + "_" + str(second_cell_id) + ".png"
plt.savefig(os.path.join(image_dir, file_name))
