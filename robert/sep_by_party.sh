#!/bin/bash 
grep "Democrat" primary_results.csv > democrat_primary_results.csv

grep "Republican" primary_results.csv | sed s/'Ben Carson'/'Other'/ | sed s/'John Kasich'/'Other'/ | sed s/'Marco Rubio'/'Other'/ | sed s/'Ted Cruz'/'Other'/ | sort -t',' -k1 -k2 -k3 -k5 -k6 > temp

python3 combine_primary_others.py temp

grep 
