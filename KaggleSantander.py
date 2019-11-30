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

corr = x_train.corr()
corr
plt.matshow(corr)
plt.title('Correlation matrix for Santander data')
plt.show()

PointBiserial = []
for i in range(200): 
	R = sc.pointbiserialr(y_train, x_train.iloc[:,i]) 
	PointBiserial.append([R.correlation, R.pvalue])
PointBiserial = pd.DataFrame(PointBiserial)
PointBiserial.columns = ['correlation','pvalue']
PointBiserial.sort_values('correlation')
Significant = PointBiserial[np.abs(PointBiserial.correlation) > 0.05]
Significant.shape

# Standarize data
mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0)
x_train = (x_train - mean)/std
test_data.iloc[:,1:201]  = (test_data.iloc[:,1:201]  - mean)/std

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

# Logistic model estimated only on Significant variables
Logit = linear_model.LogisticRegression(C = 1e42, penalty='l2', solver = 'saga', max_iter = 1000, multi_class='ovr')
Logit.fit(x_train.iloc[:,Significant.index.values], y_train)
LogitPrediction = Logit.predict(x_val.iloc[:,Significant.index.values])
Assessment(LogitPrediction)

# Lasso logistic model - predict using the logistic model with L1 penalization with alpha selected using cross-validation
LassoLogisticModel = linear_model.LogisticRegressionCV(Cs = 100, cv = 5, penalty = 'l1', solver = 'saga', max_iter = 1000, 
                                                        multi_class = 'ovr', refit = True, scoring = 'roc_auc', n_jobs = os.cpu_count() - 1) 
LassoLogisticModel.fit(x_train, y_train)
LassoLogisticModelPrediction = LassoLogisticModel.predict(x_val)
Assessment(LassoLogisticModelPrediction)

# Simplest Neural Network
from keras import models
from keras import layers
from keras import regularizers                                                                                                       
from keras import callbacks

model = models.Sequential()
model.add(layers.Dense(128, activation = 'relu', input_shape = (200,)))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))
NNPrediction = model.predict(x_val)
Assessment(NNPrediction)


def Loss(history): 
	loss = history.history['loss'] 
	val_loss = history.history['val_loss'] 
	epochs = range(1, len(loss) + 1) 
	plt.plot(epochs, loss, 'bo', label='Training loss') 
	plt.plot(epochs, val_loss, 'b', label='Validation loss') 
	plt.title('Training and validation loss') 
	plt.xlabel('Epochs') 
	plt.ylabel('Loss') 
	plt.legend() 
	plt.show() 
                                                                                                                                   

def Accuracy(history): 
	acc = history.history['acc'] 
	val_acc = history.history['val_acc']
	epochs = range(1, len(acc) + 1) 
	plt.plot(epochs, acc, 'bo', label='Training acc') 
	plt.plot(epochs, val_acc, 'b', label='Validation acc') 
	plt.title('Training and validation accuracy') 
	plt.xlabel('Epochs') 
	plt.ylabel('Loss') 
	plt.legend() 
	plt.show() 

# Very complicated densly connected NN
model = models.Sequential()                                                                                                          
model.add(layers.Dense(1024, activation = 'relu', input_shape = (200,), kernel_regularizer=regularizers.l1(0.001)))                                                            
model.add(layers.Dense(512, activation = 'relu', kernel_regularizer=regularizers.l1(0.001)))                                                                                     
model.add(layers.Dense(256, activation = 'relu', kernel_regularizer=regularizers.l1(0.001))) 
model.add(layers.Dense(128, activation = 'relu',kernel_regularizer=regularizers.l1(0.001)))                                                                             
model.add(layers.Dense(64, activation = 'relu',kernel_regularizer=regularizers.l1(0.001))) 
model.add(layers.Dense(32, activation = 'relu',kernel_regularizer=regularizers.l1(0.001)))                                                                          
model.add(layers.Dense(16, activation = 'relu', kernel_regularizer=regularizers.l1(0.001)))                                                                                    
model.add(layers.Dense(1, activation = 'sigmoid'))
callbacks_list=[callbacks.EarlyStopping(monitor='acc',patience=2,)]                                                                                   
model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy']) 
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val), callbacks=callbacks_list)
NNPrediction = model.predict(x_val)
Assessment(NNPrediction)
Accuracy(history)

# Well-balanced NN
callbacks_list=[callbacks.EarlyStopping(monitor='acc',patience=2,)]
model = models.Sequential()   
model.add(layers.Dense(512, kernel_regularizer = regularizers.l1(0.001), activation='relu', input_shape = (200,)))
model.add(layers.BatchNormalization(axis = 1)) 
model.add(layers.Dense(1, activation = 'sigmoid')) 
model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])   
model.summary()  
history = model.fit(x_train, y_train, epochs=10, batch_size=512, callbacks=callbacks_list)  
NNPrediction = model.predict(x_val)  
Assessment(NNPrediction)


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

# XGBoost
import xgboost as xgb
param = {'objective': 'binary:logistic', 'eval_metric':'auc', 'booster':'gbtree', 'verbosity':2, 'nthread':12, 'eta':0.3, 'gamma':31, 'max_depth':5, 'lambda':0.9, 'alpha':0.2, 'tree_method':'exact' }
data_xgb = xgb.DMatrix(data=train_x_data, label=train_y_data)
num_round = 200
model = xgb.train(param, data_xgb, num_round)


