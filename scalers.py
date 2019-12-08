import sklearn as skl
import xgboost as xgb
import pandas as pd
import numpy as np
import logging

logging.basicConfig(filename='./scalers.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def prepare_data(path='./train.csv'):
    data = pd.read_csv(path)
    X = data.drop(['target', 'ID_code'], 1)
    y = data.target
    prop = (y == 0).sum().astype(float) / (y == 1).sum()
    X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X, y,
                                                                          test_size=0.1, random_state=1, shuffle=False)
    return X_train, X_val, y_train, y_val, prop


def fit_and_transform(X_train, transformer):
    Transformer = transformer
    Transformer.fit(X_train)
    X_train = Transformer.transform(X_train)
    return X_train, Transformer


def validate_prediction(model, X_train, y_train, X_val, y_val, scaler):
    X_test = np.array(scaler.transform(X_val))
    fpr, tpr, thresholds = skl.metrics.roc_curve(y_train, model.predict(X_train))
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred = model.predict(X_test)
    pred = (pred > optimal_threshold).astype(int)
    print(skl.metrics.roc_auc_score(y_val, pred))
    return skl.metrics.roc_auc_score(y_val, pred)


def auc(estimator, X, y):
    prediction = estimator.predict(X)
    fpr, tpr, thresholds = skl.metrics.roc_curve(y, prediction)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred = (prediction > optimal_threshold).astype(int)
    return skl.metrics.roc_auc_score(y, pred)


def fit_model(transformer, params, grid, n_iter=100, cv=3):
    X_train, X_val, y_train, y_val, prop = prepare_data()
    X_train, Transformer = fit_and_transform(X_train, transformer)
    booster = xgb.XGBRegressor(**params, scale_pos_weight=1 / prop)
    rs = skl.model_selection.RandomizedSearchCV(cv=cv, n_jobs=1, verbose=100, scoring=auc,
                                                estimator=booster, param_distributions=grid, n_iter=n_iter,
                                                random_state=1)
    rs.fit(X_train, np.array(y_train))
    result = validate_prediction(rs.best_estimator_, X_train, y_train, X_val, y_val, transformer)
    return result, rs.best_params_, rs.best_estimator_


#######################################################################################################################

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
    logger.info('Testing transformer {}...'.format(transformer))
    try:
        res, bp, bm = fit_model(transformer, params, grid, n_iter=200, cv=3)
    except Exception as err:
        logger.exception(err)
    results.append(res)
    logger.info('Best score: {}'.format(res))
    models.append(bm)
    parameters.append(bp)

tmp = pd.DataFrame(parameters)
tmp['results'] = results
tmp['transformer'] = transformers
tmp = tmp.astype(str)
tmp.to_csv('./best_models.csv')
