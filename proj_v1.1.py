import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 

county_facts = pd.read_csv('county_facts.csv')
county_facts.set_index(county_facts['fips'])
county_facts_dictionary = open('county_facts_dictionary.csv', 'r')
lines = county_facts_dictionary.readlines()
cols = county_facts.columns.values
cols = cols[3:]
cf = county_facts[cols].copy() # A partial copy of county facts for us to modify.
for c in cols:
    c_max = county_facts[c].max()
    c_min = county_facts[c].min()
    #print(c_max)
    #print(c_min)
    cf[c] = (cf[c] - c_min)/(c_max - c_min)

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
    example = p.split(l.strip())
    trainX.append(factdict.get(example[2][:-2]))
    trainY.append(example[4])

def test_results(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  for l in f:
    #print l
    example = p.split(l.strip())
    #print example[2][:-2]
    testX.append(factdict.get(example[2][:-2]))
    testY.append(example[4])

def preprocess(X,y):
  # Verify X and y are same length.
  if(len(X) != len(y)):
    print('Sample features and results must be same length')
    sys.exit()
  # Remove entries that are NoneType
  tempX = []
  tempY = []
  for i in range(len(X)):
    if( not X[i]):
      continue
    tempX.append([float(x) for x in X[i]])
    tempY.append(float(y[i]))
  X = tempX[:]
  y = tempY[:]
  maxs = [0.0] * len(X[0])
  mins = [0.0] * len(X[0])
  for i in range(len(X)):
    for j in range(len(X[i])):
      maxs[j] = max(maxs[j], X[i][j])
      mins[j] = min(mins[j], X[i][j])
  for i in range(len(X)):
    for j in range(len(X[i])):
      X[i][j] = (X[i][j] - mins[j])/(maxs[j] - mins[j])
  X = np.array(X)
  y = np.array(y)
  return X, y

print('Parsing data')
read_param('county_facts_dictionary.csv')
read_facts('county_facts.csv')
#NOTE: temporary training data, until final is parsed.
train_results('temp_clinton_train.csv')
test_results('temp_clinton_test.csv')

trainX, trainY = preprocess(trainX,trainY)
testX, testY = preprocess(testX,testY)

print('Plotting features')
fig_index = 0
#for i in range(len(trainX[0])):
     fig_index += 1
#    plt.figure(fig_index)
#    plt.scatter(trainY,trainX[:,i], color='blue',label='Feature')
#    plt.hold('on')
#    plt.xlabel(paramdict[cols[i]])
#    plt.ylabel('Target Value')
#    plt.title('Feature Exploration:' + paramdict[cols[i]])
#    plt.legend()
#    plt.savefig('./robert/' + cols[i] + '.pdf')

#model = []
#svr = svm.SVR()
## We will use this dictionary of parameter values to test each combination 
## and find the values that optimize performance.
##parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000], \
##              'epsilon':[0.003,0.01,0.03,0.1], 'degree':[3,10,30,100], 'gamma':['auto',0.01,0.1]}
## The example parameter set below is smaller and only intended for debugging.
#parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1,10], 'epsilon':[0.01,0.1], 'degree':[3,10], 'gamma':['auto', 0.01, 0.1]}
#svr_model = GridSearchCV(svr, parameters)
#svr_model.fit(trainX,trainY)
#
##svr_model.cv_results_
#svr_model.best_score_
#svr_model.best_params_
#svr_model.best_estimator_
#svr_best_model = svr_model.best_estimator_
##print(svr_model.cv_results_['rank_test_score'])
#ms = len(svr_model.cv_results_['rank_test_score'])
#best_models_params = []
#for m in range(ms):
#    if(svr_model.cv_results_['rank_test_score'][m] <= 5):
#        best_models_params.append(svr_model.cv_results_['params'][m])
##print(svr_model.cv_results_)

print('Testing SVR models')
X_train, X_validate, y_train, y_validate = train_test_split(trainX, trainY, test_size=0.4, random_state=0)
parameters = {'C':[1e-5, 1e0, 1e5], 'gamma':[0.01, 0.1], 'epsilon':[0.03, 0.3]}
svr = svm.SVR(kernel='rbf', verbose=True)
svr_rbf = GridSearchCV(svr, parameters)
svr_rbf.fit(X_train,y_train)
svr_rbf_model = svr_rbf.best_estimator_

fig_index += 1
plt.figure(fig_index)
plt.scatter(range(0,len(y_test)),y_test,color='blue',label='Validation Targets')
plt.hold('on')
plt.plot(range(0,len(y_test)),y_test,color='red',label='Validation Targets')
plt.xlabel('Sample Index')
plt.ylabel('Clinton % of vote')
plt.title('Primary Validation Set Prediction')
plt.legend()
plt.show()
