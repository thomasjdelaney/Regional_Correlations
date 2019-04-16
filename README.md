A repository for calculating the pairwise correlations between neural firing data from many neurons in different brain regions, then clustering these correlations, then comparing this clustering to the biological partition of the neurons.

Note that the neuropixels data shared by Nick Steinmetz is the databased used here. But it is not contained within the repo, because it takes up too much space.

### correlation_matrix.py

Calculate pairwise correlations between given choice of neurons.

optional arguments:
* **-h, --help**: show the help message and exit
* **-c, --cell_choice**: 'random' or 'specified'. The method of choosing the cells.
* **-n, --number_of_cells**: Int. The number of cells to choose at random.
* **-i, --cell_ids**: List of ints corresponding to cluster_ids.  Used when cell_choice is "specified".
* **-g, --group**: Choose any combination of 'good' 'mua' 'unsorted'. The quality of sorting for randomly chosen_cells.
* **-p, --probe**: Filter the randomly chosen cells by probe ('posterior' or 'frontal').
* **-r, --region**: Filter the randomly chosen cells by region ('motor_cortex', 'striatum', 'hippocampus', 'thalamus', or 'v1').
* **-s, --numpy_seed**: The seed to use to initialise numpy.random. Determines which cells are randomly chosen.

Running the following command will choose ten good neurons at random, calculate the pairwise correlations between their spike counts during ten trials each of drifiting gratings, and present to you this pairwise correlation matrix, and a table of information about the cells.
```bash
python -i py/regional_correlations.py --cell_choice random --number_of_cells 10 --group good --numpy_seed 1798
```

### bin_width_variation.py

Find some strongly or weakly responding cells. Randomly select a given number of pairs of these cells, pairing neurons within the same region. Calculate the pairwise correlations and mutual information of these pairs across different time bin widths. Save all this down to a csv with a given filename.

optional arguments:
* **-h, --help**: show the help message and exit
* **-n, --wanted_num_pairs**: The number of strongly (or weakly) responding pairs to use.
* **-g, --group**: The quality of sorting for randomly chosen_cells.
* **-s, --numpy_seed**: The seed to use to initialise numpy.random.
* **-a, --is_strong**: Flag for strongly or weakly responding cells.
* **-f, --filename**: Name of file for saving the csv.
* **-t, --threshold**: Threshold spike count for trial classifying a cell as "strongly responding".
* **-d, --debug**: Enter debug mode.

Running the following command will find cells with a firing rate of at least 10 spikes per second. Then select 30 pairs from all the possible pairs of these cells. Then will calculate the pairwise correlation and mutual information for each of these 30 pairs across different time bins. Then will save down this information into a csv file in the csv directory.

```bash
python py/bin_width_variation.py --wanted_num_pairs 30 --group good --filename test.csv --threshold 20.0
```

### bin_width_plotting.py

Plotting results from the csv files created by ```bin_width_variation.py```. Mainly spike count correlations, signal correlations, and mutual information against bin width.

### correlation_histograms.py

For making histograms from the results in the csv files created by ```bin_width_variation.py```. Mainly spike count correlations and mutual information.

### cross_correlograms.py

For making cross correlograms for pairs of neurons.

### signal_correlations.py

For calculating the signal correlations between many neurons using different bin widths.

### statistical_tests.py

For carrying out statistical tests on the samples in the csv directory.

###### TO DO:
- update the README.md
- perform stat tests on firing rates
- geomtric mean vs pairwise correlations
- correlations vs mutual information
- community detection
- population coupling (Okun)
- big correlation matrices
