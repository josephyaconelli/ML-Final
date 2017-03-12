import re
import os
import sys

with open('../random_states.csv', 'r') as rs_f:
    random_states = rs_f.readline()
    random_states = random_states.strip('\n').split(',')
    print(random_states)
    
with open('../primary_results.csv', 'r') as pr_f:
    line = pr_f.readline()
    l = line.strip('\n').split(',')
    labels = l
    line = pr_f.readline()
    l = line.strip('\n').split(',')
    print(l)
    lines = pr_f.readlines()
    with open('../clinton_train.csv', 'w') as ct_f:
        with open('../trump_train.csv', 'w') as tt_f:
            for line in lines:
                l = line.strip('\n').split(',')
                if(l[5] == 'Hillary Clinton'):
                    ct_f.write(l[0] + ',' + l[2] + ',' + l[3] + ',' + l[6] + ',' + l[7] + '\n')
                elif(l[5] == 'Donald Trump'):
                    tt_f.write(l[0] + ',' + l[2] + ',' + l[3] + ',' + l[6] + ',' + l[7] + '\n')

  
        
