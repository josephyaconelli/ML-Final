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

def separate_data(data):
  # Separating data into list of feature vectors and target vector.
  X = []
  y = []
  for r in data:
      #print(r)
      x = r[0]
      if(not x):
          continue
      if(len(x) != 51):
          print('This datapoint is a different length, check before proceeding.')
          sys.exit()
      # Only look at 10 features for now. 
      x = x[:4]
      X.append(x)
      y.append(r[1])
  X = np.array(X)
  y = np.array(y)
  return X,y
  
def explore_data(X,y):
  # We will use our judgement to do some pre-selection for important features.
  # PST040214: Population, 2014 estimate
  # PSTPersons 65 years and over, percent
  print(' ')

def train_model(X, y):
  # Defining model with linear kernel.
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto', epsilon=0.1, verbose=True)
  svr_lin = SVR(kernel='linear', C=1e3, epsilon=0.1, verbose=True, cache_size=7000)
  svr_poly = SVR(kernel='poly', C=1e1, degree=10, max_iter=1000000, epsilon=0.4, verbose=True)
  #print(np.shape(X))
  #print(X)
  svr_rbf_model = svr_rbf.fit(X,y)
  svr_lin_model = svr_lin.fit(X,y)
  svr_poly_model = svr_poly.fit(X,y)
  rbf_preds = svr_rbf_model.predict(X)
  lin_preds = svr_lin_model.predict(X)
  poly_preds = svr_poly_model.predict(X)
  # View how well the model can predict the training data as a sanity check.
  print(rbf_preds)
  print(lin_preds)
  print(poly_preds)
  lw = 1
  plt.scatter(range(0,len(y)),y, color='red', label='data')
  plt.hold('on')
  plt.plot(range(0,len(y)),rbf_preds, color='blue', lw=lw, label='RBF model')
  plt.plot(range(0,len(y)),lin_preds, color='green', lw=lw, label='Linear model')
  plt.xlabel('sample index')
  plt.ylabel('target')
  plt.title('Primary Training Data Support Vector Regression')
  plt.legend()
  plt.show()
  print('Classifier training complete')
  return svr_rbf_model, svr_lin_model, svr_poly_model

def test_model(X, y, model):
  p = model.predict(X)
  for i in range(len(y)):
    print('Truth: ' + str(y[i]) + ', prediction: ' + str(p[i]))
  return p

def view_results(results):
   lw = 2
   plt.hold('on')
 
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

  # Separate X matrix and y vector for training and test data.
  xTrain, yTrain = separate_data(train_data)
  xTest, yTest = separate_data(test_data)

  # Exploration section, plot various features against the to see what type of
  # feature scaling is appropriate.
  #explore_data(xTrain, yTrain)
 
  svr_rbf_model, svr_lin_model, svr_poly_model = train_model(xTrain,yTrain)
  results = test_model(xTest, yTest, svr_rbf_model)
  results = test_model(xTest, yTest, svr_lin_model)
  results = test_model(xTest, yTest, svr_poly_model)
  view_results(results)
  #reg = linear_model.Ridge (alpha = .5)



if __name__ == "__main__":
  main(sys.argv[1:])
