import sys
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


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
    example = p.split(l.strip())
    trainX.append(factdict.get(example[2][:-2]))
    trainY.append(example[4])
  return [(x,y) for x, y in zip(trainX,trainY)]


def obs_acc_freq(target, predictions, box_size=0.025):
	pred_labels = [predictions[i][1] for i in range(len(predictions))]
	predictions = [predictions[i][0] for i in range(len(predictions))]
	for i in range(len(predictions)):
		if(len(target) != len(predictions[i])):
			print('Error, target vector and prediction vector have different lengths')
			return None
	boxes = [[(round(box_size* i,3), round(box_size*(i+1),3)),[0]*len(predictions),0] for i in range(math.ceil(1.0/box_size))]
	for i in range(len(target)):
		for j in range(len(predictions)):
			l1_error = abs(float(target[i]) - float(predictions[j][i]))
			print("Prediction: ", predictions[j][i])
			if l1_error >= 1.0:
				l1_error = 0.99
			assignment = math.floor(l1_error / box_size)
			print("Assignment: ", assignment)
			boxes[assignment][1][j] += 1
		bl_l1_error = abs(float(target[i]) - 0.5)
		assignment = math.floor(bl_l1_error / box_size)
		boxes[assignment][2] += 1
	c = ['Dist True', 'Baseline']
	for i in range(len(pred_labels)):
		c.append(pred_labels[i])
	for i in range(len(boxes)):
		temp = boxes[i]
		boxes[i] = [boxes[i][0], boxes[i][2]]
		for j in range(len(predictions)):
			boxes[i].append(temp[1][j])
	df = pd.DataFrame(boxes, columns=c)
	df = df.set_index(c[0])
	return df



def calcAccuracy(y_test, predictions):
  freq_table = obs_acc_freq(y_test, predictions)
  print('Total Predicted Counties: ' + str(len(y_test)))
  #print(freq_table)
  freq_table = freq_table.drop(freq_table.index[range(23,40)])
  plt.figure(1,figsize=(25,5))
  ax = freq_table.plot(kind='bar', figsize=(25,5),)
  ax.set_xlabel('Accuracy Range')
  ax.set_ylabel("Frequency")
  plt.show()


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
      # Using 
      features_chosen = [0,6,7,8,9,10,15,18,20,21,22,24,25,29,30,31,32,47,49,50]
      x = [x[i] for i in features_chosen]
      X.append(x)
      y.append(r[1])
  X = np.array(X)
  y = np.array(y)
  return X,y

def preprocess(X,y):
  maxs = [0.0] * len(X[0])
  mins = [0.0] * len(X[0])
  for i in range(len(X)):
    for j in range(len(X[i])):
      maxs[j] = max(maxs[j], X[i][j])
      mins[j] = min(mins[j], X[i][j])
  for i in range(len(X)):
    for j in range(len(X[i])):
      X[i][j] = (X[i][j] - mins[j])/(maxs[j] - mins[j])
    
  
def explore_data(X,y):
  # We will use our judgement to do some pre-selection for important features.
  # PST040214: Population, 2014 estimate
  # PSTPersons 65 years and over, percent
  print(' ')

def tune_params(X, y, model, tuned_params):
  pass

def train_model(X, y, models):
  if('logistic_regression' in models):
    logreg = linear_model.LogisticRegression(C=1e5)
  
def ensemble_predict(parts):
  tuning = [.1, .2, .7]
  print("Ensemble stuff")
  print(parts[0])
  print("There is was!")
  en_pred = [0.0]*len(parts[0])
  for i in range(len(parts)):
    print(i)
    print(tuning[i])
    for j in range(len(parts[i])):
      print(j)
      en_pred[j] += float(parts[i][j])*tuning[i]

  for i in range(len(en_pred)):
    en_pred[i] = en_pred[i]

  print("Ensemble predictions:")
  print(en_pred)
  print("Ensemble Predictions END")

  return en_pred

def train_models(X, y):
  # Defining model with linear kernel.
  logreg = linear_model.LogisticRegression(C=1e10)
  ridge = Ridge(alpha=1.0)
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto', epsilon=0.1, verbose=True)
  svr_lin = SVR(kernel='linear', C=1e1, epsilon=0.01, verbose=True, cache_size=7000)
  svr_poly = SVR(kernel='poly', C=1e2, degree=7, epsilon=0.005, verbose=True)
  #print(np.shape(X))
  print(X)
  logreg_model = logreg.fit(X,y)
  print('Logistic Regression Training Complete.')
  #ridge_model = ridge.fit(X,y)
  print('Ridge Regression Training Complete.')
  svr_rbf_model = svr_rbf.fit(X,y)
  print('SVR RBF Training Complete')
  svr_lin_model = svr_lin.fit(X,y)
  print('SVR Linear Training Complete')
  svr_poly_model = svr_poly.fit(X,y)
  print('SVR Polynomial Training Complete')
  print(logreg_model.score(X,y))
  #print(svr_rbf_model.score(X,y))
  #print(svr_lin_model.score(X,y))
  #print(svr_poly_model.score(X,y)
  print('RBF, weights')
  #print(svr_rbf_model.coef_)
  #print(svr_rbf_model.intercept_)
  print('Linear, weights')
  print(svr_lin_model.coef_)
  print(svr_lin_model.intercept_)
  print('Polynomial, weights')
  #print(svr_poly_model.coef_)
  #print(svr_poly_model.intercept_)
  logreg_preds = logreg_model.predict(X)
  rbf_preds = svr_rbf_model.predict(X)
  lin_preds = svr_lin_model.predict(X)
  poly_preds = svr_poly_model.predict(X)
  
  ensemble_preds = ensemble_predict([poly_preds, rbf_preds, lin_preds])

  # View how well the model can predict the training data as a sanity check.
  print(rbf_preds)
  print(lin_preds)
  print(poly_preds)
  lw = 1
  plt.scatter(range(0,len(y)),y, color='red', label='Test Data')
  plt.hold('on')
  plt.plot(range(0,len(y)),rbf_preds, color='blue', lw=lw, label='RBF model')
  plt.plot(range(0,len(y)),lin_preds, color='green', lw=lw, label='Linear model')
  plt.plot(range(0,len(y)),poly_preds, color='purple', lw=lw, label='Poly model')
  plt.plot(range(0,len(y)),ensemble_preds, color='orange', lw=lw, label='Ensemble model')
  plt.plot(range(0,len(y)), logreg_preds, color='black', lw=lw, label='Logistic Regression')
  plt.xlabel('sample index')
  plt.ylabel('target')
  plt.title('Primary Training Data Support Vector Regression')
  plt.legend()
  plt.show()
  
  predictions = []
  
  predictions.append([rbf_preds, "RBF"])
  predictions.append([logreg_preds, "Logistic Regression"])
  predictions.append([lin_preds, "Linear Regression"])
  predictions.append([poly_preds, "Polynomial"])
  predictions.append([ensemble_preds, "Ensemble"])
  
  calcAccuracy(y, predictions)
  print('Classifier training complete')

  return logreg_model, svr_rbf_model, svr_lin_model, svr_poly_model

def test_model(X, y, model):
  p = model.predict(X)
  for i in range(len(y)):
    print('Truth: ' + str(y[i]) + ', prediction: ' + str(p[i]))
  lw = 2
  plt.scatter(range(0,len(y)),y,color='red',label='Test Data')
  plt.hold('on')
  plt.plot(range(0,len(y)),p,color='blue',lw=lw,label='Model Predictions')
  plt.xlabel('Sample Index')
  plt.ylabel('Target')
  plt.title('Primary Test Data Predictions')
  plt.legend()
  plt.show()
  return p

def test_ensemble(X, y, logreg_model, svr_rbf_model, svr_lin_model):
  logreg_preds = logreg_model.predict(X)
  rbf_preds = svr_rbf_model.predict(X)
  lin_preds = svr_lin_model.predict(X)
  # poly_preds = svr_poly_model.predict(X)
  
  p = ensemble_predict([lin_preds, rbf_preds, logreg_preds])

  for i in range(len(y)):
    print('Truth: ' + str(y[i]) + ', prediction: ' + str(p[i]))
  lw = 2
  plt.scatter(range(0,len(y)),y,color='red',label='Test Data')
  plt.hold('on')
  plt.plot(range(0,len(y)),p,color='blue',lw=lw,label='Model Predictions')
  plt.xlabel('Sample Index')
  plt.ylabel('Target')
  plt.title('Primary Test Data Predictions')
  plt.legend()
  plt.show()
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
  preprocess(xTrain,yTrain)
  preprocess(xTest,yTest)
  

  # Exploration section, plot various features against the to see what type of
  # feature scaling is appropriate.
  #explore_data(xTrain, yTrain)
  print(xTest)
  print(yTest)
 
  logreg_model, svr_rbf_model, svr_lin_model, svr_poly_model = train_models(xTrain,yTrain)
  rbf_preds = test_model(xTest, yTest, svr_rbf_model)
  lin_preds = test_model(xTest, yTest, svr_lin_model)
  poly_preds = test_model(xTest, yTest, svr_poly_model)
  logreg_preds = test_model(xTest,yTest, logreg_model)
  #ridge_preds = test_model(xTest,yTest, ridge_model)
  ensemble_preds = test_ensemble(xTest, yTest, svr_poly_model, svr_rbf_model, svr_lin_model)
  #view_results(results)

  
  predictions = []
  
  predictions.append([rbf_preds, "RBF"])
  predictions.append([logreg_preds, "Logistic Regression"])
  predictions.append([lin_preds, "Linear Regression"])
  predictions.append([poly_preds, "Polynomial"])
  predictions.append([ensemble_preds, "Ensemble"])
  
  calcAccuracy(yTest, predictions)

  #reg = linear_model.Ridge (alpha = .5)



if __name__ == "__main__":
  main(sys.argv[1:])
