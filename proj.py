import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  print varnames
  namehash = {}
  for l in f:
    example = [x for x in p.split(l.strip())]
    #print x[0]
    #print x[1]
    #print x[2]
    #print x[3]
    #x = example[0:-1]
    #y = example[-1]
    # Each example is a tuple containing both x (vector) and y (int)
    xdata = []
    ydata = []
    #xdata.append(x)
    #ydata.append(y)
  return (xdata, ydata, varnames)
  
  # Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  # Process command line arguments.
  # (You shouldn't need to change this.)
  if (len(argv) != 3):
    print 'Usage: perceptron.py <train> <test> <model>'
    sys.exit(2)
  (trainx, trainy, varnames) = read_data(argv[0])
  (testx, testy, testvarnames) = read_data(argv[1])
  #modelfile = argv[2]
  #x = np.array(trainx)
  #y = np.array(trainy)
  #print trainx[0]
  #print trainy[0]
  reg = linear_model.Ridge (alpha = .5)



if __name__ == "__main__":
  main(sys.argv[1:])

