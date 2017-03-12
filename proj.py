
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    # Each example is a tuple containing both x (vector) and y (int)
    data.append( (x,y) )
  return (data, varnames)
  
  # Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  # Process command line arguments.
  # (You shouldn't need to change this.)
  if (len(argv) != 3):
    print 'Usage: perceptron.py <train> <test> <model>'
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  modelfile = argv[2]
  ytemp = []
  for instance in train:
    ytemp.append(instance.pop(len(varnames)))
  x = np.array(train)
  y = np.array(ytemp)
  print train[2]
  print y[0]
  print y[1]
  print y[2]
  print y[3]
  reg = linear_model.Ridge (alpha = .5)
