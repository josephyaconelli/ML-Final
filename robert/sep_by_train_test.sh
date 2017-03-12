grep -E 'Rhode Island|New York|Maryland|Tennessee|Ohio|Texas|Missouri|Colorado|Nevada|Oregon' ../democrat_primary_results.csv > ../democrat_test.csv

grep -v -E 'Rhode Island|New York|Maryland|Tennessee|Ohio|Texas|Missouri|Colorado|Nevada|Oregon' ../democrat_primary_results.csv > ../democrat_train.csv

grep -E 'Rhode Island|New York|Maryland|Tennessee|Ohio|Texas|Missouri|Colorado|Nevada|Oregon' ../republican_primary_results.csv > ../republican_test.csv

grep -v -E 'Rhode Island|New York|Maryland|Tennessee|Ohio|Texas|Missouri|Colorado|Nevada|Oregon' ../republican_primary_results.csv > ../republican_train.csv
