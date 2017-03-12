import sys
import os


republican_results = sys.argv[1]
print('test')
with open(republican_results, "r") as f:
    with open("republican_primary_results.csv", "w") as g:
        line = f.readline()
        l = line.split(',')
        state = l[0]
        code = l[1]
        county = l[2]
        val = l[3]
        party = l[4]
        candidate = l[5]
        votes = int(l[6])
        per = float(l[7])
        lines = f.readlines()
        for line in lines:
            l = line.split(',')
            if(l[0] == state and l[1] == code and l[2] == county and l[3]  == val and l[4] == party and l[5] == candidate):
                votes += int(l[6])
                per += float(l[7])
            else:
                g_line = state + ',' + code + ',' + county + ',' + str(val) + ',' + party + ',' + candidate + ',' + str(votes) + ',' + str(per) + '\n'
                g.write(g_line)
                state = l[0]
                code = l[1]
                county = l[2]
                val = l[3]
                party = l[4]
                candidate = l[5]
                votes = int(l[6])
                per = float(l[7])
            
      
        
