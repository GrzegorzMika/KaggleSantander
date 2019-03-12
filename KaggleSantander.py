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

# Split data into targets and predictors
y_train = train_data['target']
x_train = train_data.drop(columns = ['ID_code', 'target'])

# Inspect data
train_data.head()
train_data.shape
train_data.memory_usage()
train_data.columns
train_data.groupby(by = 'target').count()
train_data.describe()
kur = sc.kurtosis(x_train)
pd.DataFrame(kur).describe()
skew = sc.skew(x_train) 
pd.DataFrame(skew).describe()
for i in range(200): 
	plt.subplot(40, 5, i + 1) 
	x_train.iloc[:,i].plot(kind = 'density') 
AndersonDarling = [] 
for i in range(200): 
	AndersonDarling.append(sc.anderson(x_train.iloc[:,i]).statistic)
pd.DataFrame(AndersonDarling).describe()

# Standarize data
mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0)
x_train = (x_train - mean)/std

# Split data into training and testing set
train_x_data = x_train
train_y_data = y_train
x_train, x_val, y_train, y_val = train_test_split(train_x_data, train_y_data, test_size = 0.2)
x_train.shape
x_val.shape

def Assessment(pred, y = y_val):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    auc = metrics.auc(fpr, tpr)
    print(auc)
    return auc

# Simplest approach - predict always the more common value in training set
SimplestApproachPrediction = [0] * x_val.shape[0]
Assessment(SimplestApproachPrediction)

# Random approach - predict completly at random
RandomApproachPrediction =np.round(np.random.rand(y_val.shape[0]))
Assessment(RandomApproachPrediction)


# Logistic model - predict using logistic model without any penalization
LogisticModel = linear_model.LogisticRegression(C = 1e42, penalty='l2', solver = 'saga', max_iter = 1000, multi_class='ovr')
LogisticModel.fit(x_train, y_train)
LogisticModelPrediction = LogisticModel.predict(x_val)
Assessment(LogisticModelPrediction)

# Lasso logistic model - predict using the logistic model with L1 penalization with alpha selected using cross-validation
LassoLogisticModel = linear_model.LogisticRegressionCV(Cs = 100, cv = 5, penalty = 'l1', solver = 'saga', max_iter = 1000, 
                                                        multi_class = 'ovr', refit = True, scoring = 'roc_auc', n_jobs = os.cpu_count() - 1) 
LassoLogisticModel.fit(x_train, y_train)
LassoLogisticModelPrediction = LassoLogisticModel.predict(x_val)
Assessment(LassoLogisticModelPrediction)












