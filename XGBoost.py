import os 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn import linear_model 
import scipy.stats as sc 
import matplotlib.pyplot as plt 
 
# Read data 
data_dir = '/home/grzegorz/Kaggle/Santander' 
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv')) 
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv')) 
 
# Split data into targets and predictors 
y_train = train_data['target'] 
x_train = train_data.drop(columns = ['ID_code', 'target'])
mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0)
x_train = (x_train - mean)/std
test_data.iloc[:,1:201]  = (test_data.iloc[:,1:201]  - mean)/std 
train_x_data = x_train 
train_y_data = y_train 
x_train, x_val, y_train, y_val = train_test_split(train_x_data, train_y_data, test_size = 0.2) 

def Assessment(pred, y = y_val): 
	fpr, tpr, thresholds = metrics.roc_curve(y, pred) 
	auc = metrics.auc(fpr, tpr) 
	print(auc) 
	return auc 
  
def SelectTreshold(model, x = x_train, y = y_train): 
	pred = model.predict(x) 
	fpr, tpr, thresholds = metrics.roc_curve(y, pred) 
	sens = tpr + (1 - fpr) 
	best_id = np.argmax(sens) 
	treshold = thresholds[best_id] 
	return(treshold) 

def MakePrediction(model, x = x_val, y = y_val, train_x = x_train, train_y = y_train): 
	threshold = SelectTreshold(model, train_x, train_y) 
	pred = model.predict(x_val) 
	pred = pred > threshold 
	pred = pred.astype('float32') 
	print(Assessment(pred, y_val)) 
	return(pred) 
 
def TresholdAgain(pred, threshold): 
	pred = pred > threshold 
	pred = pred.astype('float32') 
	auc = Assessment(pred, y_val) 
	return(auc) 

import xgboost as xgb

data_xgb = xgb.DMatrix(data=x_train, label=y_train)
x_val = xgb.DMatrix(x_val)
num_round = 200

width = []
for i in np.arange(0,1,0.02): 
	param = {'objective': 'binary:logistic', 'eval_metric':'auc', 'booster':'gbtree', 'verbosity':2, 'nthread':12, 'eta':i, 'gamma':10,'max_depth':5, 'lambda':0.1, 'alpha':0.1, 'tree_method':'exact' }  
	model = xgb.train(param, data_xgb, num_round)  
	width.append(Assessment(model.predict(x_val)))
l1 = []
for i in np.arange(0,1,0.02): 
	param = {'objective': 'binary:logistic', 'eval_metric':'auc', 'booster':'gbtree', 'verbosity':2, 'nthread':12, 'eta':0.3, 'gamma':10,'max_depth':5, 'lambda':0, 'alpha':i, 'tree_method':'exact' }  
	model = xgb.train(param, data_xgb, num_round)  
	l1.append(Assessment(model.predict(x_val)))
l2 = []
for i in np.arange(0,1,0.02): 
	param = {'objective': 'binary:logistic', 'eval_metric':'auc', 'booster':'gbtree', 'verbosity':2, 'nthread':12, 'eta':0.3, 'gamma':10,'max_depth':5, 'lambda':i, 'alpha':0, 'tree_method':'exact' }  
	model = xgb.train(param, data_xgb, num_round)  
	l2.append(Assessment(model.predict(x_val)))
gamma = []
for i in range(100): 
	param = {'objective': 'binary:logistic', 'eval_metric':'auc', 'booster':'gbtree', 'verbosity':2, 'nthread':12, 'eta':0.3, 'gamma':i,'max_depth':5, 'lambda':0.1, 'alpha':0.1, 'tree_method':'exact' }  
	model = xgb.train(param, data_xgb, num_round)  
	gamma.append(Assessment(model.predict(x_val)))
