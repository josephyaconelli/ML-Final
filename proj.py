import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

paramdict = {}
factdict = {}


def read_facts(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  f.readline()
  f.readline()
  count = 0
  for l in f:
    value = p.split(l.strip())
    key = value.pop(0)
    #print key
    #print len(value[0])
    if len(value[1]) < 1:
      count += 1
      continue
    value.pop(0)
    value.pop(0)
    factdict[key] = [float(x) for x in value]
    
def read_param(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  f.readline()
  for l in f: 
    list = p.split(l.strip())
    paramdict[list[0]] = list[1]

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
  if (len(argv) != 5):
    print 'Usage: perceptron.py <train> <test> <facts> <dictionary> <model>'
    sys.exit(2)
  read_param(argv[3])
  #(trainx, trainy, varnames) = read_data(argv[0])
  #(testx, testy, testvarnames) = read_data(argv[1])
  #modelfile = argv[2]
  #x = np.array(trainx)
  #y = np.array(trainy)
  #print trainx[0]
  #print trainy[0]
  read_facts(argv[2])
  reg = linear_model.Ridge (alpha = .5)



if __name__ == "__main__":
  main(sys.argv[1:])

