import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, model_selection
import xgboost as xgb
import logging
import sys

logging.basicConfig(filename='./log.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def auc(estimator, X, y):
    prediction = estimator.predict(X)
    fpr, tpr, thresholds = metrics.roc_curve(y, prediction)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred = (prediction > optimal_threshold).astype(int)
    return metrics.roc_auc_score(y, pred)


test_train = 0.1
quantiles = 20000
n_iter = 100
cv = 3

data = './train.csv'
data = pd.read_csv(data)
logger.info("Data shape: {}".format(data.shape))

X = data.drop(['target', 'ID_code'], 1)
y = data.target
X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=test_train)
logger.info('Test-train split {}'.format(test_train))

prop = (y == 0).sum().astype(float) / (y == 1).sum()
logger.info('Positive vs negative group proportion {}'.format(1. / prop))

QuantileScaler = preprocessing.QuantileTransformer(n_quantiles=quantiles)
QuantileScaler.fit(X_train)
X_train = QuantileScaler.transform(X_train)
logger.info('Number of quantiles used for scaling: {}'.format(quantiles))

params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_jobs': 12, 'tree_method': 'gpu_hist',
          'verbosity': 1, 'booster': 'gbtree', 'scale_pos_weight': 1 / prop,
          }
booster = xgb.XGBRegressor(**params)
logger.info('Based params: {}'.format(params))

grid = {'max_bin': [100, 250, 1000],
        'grow_policy': ['lossguide', 'depthwise'],
        'max_leaves': [0, 10, 100],
        'alpha': np.exp(np.linspace(-10, 10, 10)),
        'lambda': np.exp(np.linspace(-10, 10, 10)),
        'colsample_bytree': [0.2, 0.5, 0.8, 1],
        'colsample_bylevel': [0.2, 0.5, 0.8, 1],
        'colsample_bynode': [0.2, 0.5, 0.8, 1],
        'subsample': [0.2, 0.5, 0.8, 1],
        'max_delta_step': [0, 1, 10],
        'min_child_weight': [1, 5, 10, 100],
        'max_depth': [3, 5, 16, 50, 100],
        'gamma': np.exp(np.linspace(-10, 10, 10)),
        'eta': np.exp(np.linspace(-5, 0, 10)),
        'n_estimators': [100, 500, 1000, 5000],
        'learning_rate': [0.01, 0.1]}
logger.info('Grid: {}'.format(grid))

rs = model_selection.RandomizedSearchCV(cv=cv, n_jobs=1, verbose=100, scoring=auc,
                                        estimator=booster, param_distributions=grid, n_iter=n_iter)
logger.info('Number of CV splits: {}'.format(cv))
logger.info('Number of iterations: {}'.format(n_iter))
logger.info('Start training...')

try:
    rs.fit(X_train, np.array(y_train))
except Exception as err:
    logger.exception(err)

pd.DataFrame([rs.best_params_]).transpose().to_csv('./best_paramters.csv')
