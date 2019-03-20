#!/bin/bash

proj_dir=$HOME/Regional_Correlations
py_dir=$proj_dir/py
csv_dir=$proj_dir/csv

one_run_file=$csv_dir/correlations_by_bin_length.csv
collection_file=$csv_dir/corr_vs_width_collection.csv

for seed in {1..10}
do
  for region in motor_cortex striatum hippocampus thalamus v1
  do
    for bin_length in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
    do
      /usr/bin/python $py_dir/regional_correlations.py --number_of_cells 30 --group good --region $region --bin_length $bin_length --save_correlation_with_bin_length --stim_id 16 --numpy_seed "$seed"0
    done
    /usr/bin/python $py_dir/plot_correlations_by_bin_width.py
    if [ -f $one_run_file ]
    then
      if [ -f $collection_file ]
      then
        tail -n +2  "$one_run_file" >>  "$collection_file"
        rm $one_run_file
      else
        mv $one_run_file $collection_file
      fi
    fi
  done
done
