#!/bin/bash

proj_dir=$HOME/Regional_Correlations
py_dir=$proj_dir/py
csv_dir=$proj_dir/csv

rm $csv_dir/correlations_by_bin_length.csv

for bin_length in 2.0 1.0 0.5 0.4 0.25 0.2 0.1 0.05
do
  /usr/bin/python $py_dir/regional_correlations.py -n 20 -g good -r thalamus -b $bin_length -e -j 2
done
