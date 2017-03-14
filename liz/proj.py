import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

paramdict = {} #dictionary so we can print actual parameter titles, not abbreviations
factdict = {} #keeps track of parameters associated with a county ID number
trainX = [] #list of X vectors of the training data
trainY = [] #list of Y output of the training data
testX = [] #list of X vectors of the testing data
testY = [] #list of Y output of the testing data


def read_facts(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  f.readline()
  f.readline()
  for l in f:
    value = p.split(l.strip())
    key = value.pop(0)
    #print key
    if len(value[1]) < 1: #gets rid of rows for just states
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

def train_results(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  for l in f:
    #print l
    example = p.split(l.strip())
    #print example[2]  #float form
    trainX.append(factdict.get(example[2][:-2]))
    trainY.append(example[4])

def test_results(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  for l in f:
    #print l
    example = p.split(l.strip())
    #print example[2][:-2]
    testX.append(example[4])
    testY.append(factdict.get(example[2][:-2]))

def main(argv):
  if (len(argv) != 5):
    print 'Usage: perceptron.py <train> <test> <facts> <dictionary> <model>'
    sys.exit(2)
  read_param(argv[3])
  read_facts(argv[2])
  train_results(argv[0])
  test_results(argv[0])
  #read_results(argv[1], testX, testY)
  #print trainX[0]
  #print trainY[0]
  reg = linear_model.Ridge (alpha = .5)



if __name__ == "__main__":
  main(sys.argv[1:])
