A repository for calculating the pairwise correlations between neural firing data from many neurons in different brain regions, then clustering these correlations, then comparing this clustering to the biological partition of the neurons.

Note that the neuropixels data shared by Nick Steinmetz is the databased used here. But it is not contained within the repo, because it takes up too much space.

#### regional_correlations.py

Calculate pairwise correlations between given choice of neurons.

optional arguments:
* **-h, --help**: show the help message and exit
* **-c, --cell_choice**: 'random' or 'specified'. The method of choosing the cells.
* **-n, --number_of_cells**: Int. The number of cells to choose at random.
* **-i, --cell_ids**: List of ints corresponding to cluster_ids.  Used when cell_choice is "specified".
* **-g, --cell_group**: Choose any combination of 'good' 'mua' 'unsorted'. The quality of sorting for randomly chosen_cells.
* **-s, --numpy_seed**: The seed to use to initialise numpy.random. Determines which cells are randomly chosen.

Running the following command will choose ten good neurons at random, calculate the pairwise correlations between their spike counts during ten trials each of drifiting gratings, and present to you this pairwise correlation matrix, and a table of information about the cells.
```bash
python -i py/regional_correlations.py --cell_choice random --number_of_cells 10 --cell_group good
```
