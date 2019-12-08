import sklearn as skl
import pandas as pd
import numpy as np

import train_xgboost as txgb

transformers = [skl.preprocessing.MaxAbsScaler(), skl.preprocessing.MinMaxScaler(), skl.preprocessing.Normalizer(),
                skl.preprocessing.StandardScaler(), skl.preprocessing.RobustScaler(),
                skl.preprocessing.QuantileTransformer(), skl.preprocessing.FunctionTransformer()]

params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_jobs': 12, 'tree_method': 'hist',
          'verbosity': 1, 'booster': 'gbtree',
          }

grid = {'alpha': np.exp(np.linspace(-10, 10, 10)),
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
        'n_estimators': [100, 300, 1000]
        }
results = []
models = []
parameters = []

for transformer in transformers:
    xgb = txgb.train_xgboost(target='target', columns_to_drop=['target', 'ID_code'], params=params,
                             transformer=transformer, grid=grid, random_state=1)
    res, bp, _ = xgb.tune_model()
    results.append(res)
    parameters.append(bp)

tmp = pd.DataFrame(parameters)
tmp['results'] = results
tmp['transformer'] = transformers
tmp = tmp.astype(str)
tmp.to_csv('./best_models.csv')