import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

regions = ['motor_cortex', 'striatum', 'hippocampus', 'thalamus', 'v1']
colours = cm.gist_rainbow(np.linspace(0, 1, 5)) # 5 regions
region_to_colour = dict(list(zip(regions, colours)))

def plotCorrMatrix(corr_matrix, cell_info, region_sorted_cell_ids):
    plt.matshow(corr_matrix)
    cell_range = np.arange(corr_matrix.shape[0])
    cell_regions = cell_info.loc[region_sorted_cell_ids]['region'].values
    plt.xticks(cell_range, cell_regions)
    plt.yticks(cell_range, cell_regions)
    plt.setp(plt.xticks()[1], rotation=-45, ha="right", rotation_mode="anchor")
    plt.colorbar()
    plt.tight_layout()
