import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR

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
  return factdict
    
def read_param(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  f.readline()
  for l in f: 
    list = p.split(l.strip())
    paramdict[list[0]] = list[1]
  return paramdict

def train_results(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  for l in f:
    #print l
    example = p.split(l.strip())
    #print example[2]  #float form
    trainX.append(factdict.get(example[2][:-2]))
    trainY.append(example[4])
  return [(x,y) for x, y in zip(trainX,trainY)]

def test_results(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  for l in f:
    #print l
    example = p.split(l.strip())
    #print example[2][:-2]
    testX.append(factdict.get(example[2][:-2]))
    testY.append(example[4])
  return [(x,y) for x,y in zip(testX,testY)]

def train_model(data):
  # Separating data into list of feature vectors and target vector.
  X = []
  y = []
  for r in data:
      print(r)
      x = r[0]
      if(not x):
          continue
      if(len(x) != 51):
          print('This datapoint is a different length, check before proceeding.')
          sys.exit()
      X.append(x)
      y.append(r[1])
      #if(len(X) == 100):
      #    break
  X = np.array(X)
  # Defining model with linear kernel.
  svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1)
  svr_lin = SVR(kernel='linear', C=1e3)
  svr_poly = SVR(kernel='poly', C=1e3, degree=51)
  print(np.shape(X))
  print(X)
  svr_rbf_clf = svr_rbf.fit(X,y)
  #svr_lin_clf = svr_lin.fit(X,y)
  print('Classifier training complete')
  #return svr_lin_clf
  return svr_rbf_clf

def test_model(test_data, svr_rbf_clf):
  print(test_data)
  x = []
  for td in test_data:
    if(not td[0]):
      continue # This sample is missing (most likely because there was not a county to go with the primary result).
    x.append(td[0])
  x = np.array(x)
  p = svr_rbf_clf.predict(x)
  #print('Truth: ' + str(y) + ', prediction: ' + str(p))
  print(p)
 
def main(argv):
  if (len(argv) != 5):
    print('Usage: perceptron.py <train> <test> <facts> <dictionary> <model>')
    sys.exit(2)
  paramdict = read_param(argv[3])
  factdict = read_facts(argv[2])
  print(argv[0])
  train_data = train_results(argv[0])
  test_data = test_results(argv[1])
  #read_results(argv[1], testX, testY)
  #print trainX[0]
  #print trainY[0]
  #print(factdict)
  #print(paramdict)
  #print(train_data)
  svr_rbf_clf = train_model(train_data)
  test_model(test_data, svr_rbf_clf)
  
  reg = linear_model.Ridge (alpha = .5)



if __name__ == "__main__":
  main(sys.argv[1:])
