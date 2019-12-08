import pandas as pd
import numpy as np
import sklearn as skl
import logging
import xgboost as xgb
import multiprocessing


class train_xgboost():
    def __init__(self, target, columns_to_drop, params, transformer=skl.preprocessing.FunctionTransformer(),
                 grid=None, test_train=0.1, cv=3, niter=100, path='./train.csv', random_state=None, n_jobs=12):
        self.transformer = transformer
        self.cv = cv
        self.niter = niter
        self.path = path
        self.target = target
        self.columns_to_drop = columns_to_drop
        self.test_train = test_train
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.params = params
        self.grid = grid
        logging.basicConfig(filename='./scalers.log', level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.transformer)

    @staticmethod
    def auc(estimator, X, y):
        prediction = estimator.predict(X)
        fpr, tpr, thresholds = skl.metrics.roc_curve(y, prediction)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        pred = (prediction > optimal_threshold).astype(int)
        return skl.metrics.roc_auc_score(y, pred)

    def fit_transformer(self, X_train):
        Transformer = self.transformer
        try:
            Transformer.fit(X_train)
            X_train = Transformer.transform(X_train)
        except Exception as err:
            self.logger.exception(err)
        return X_train, Transformer

    def prepare_tune_data(self):
        try:
            data = pd.read_csv(self.path)
            X = data.drop(self.columns_to_drop, 1)
            y = data[self.target]
            prop = (y == 0).sum().astype(float) / (y == 1).sum()
            if self.random_state is not None:
                X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X, y, test_size=self.test_train,
                                                                                      random_state=self.random_state,
                                                                                      shuffle=False)
            else:
                X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X, y, test_size=self.test_train,
                                                                                      random_state=self.random_state,
                                                                                      shuffle=True)
        except Exception as err:
            self.logger.exception(err)
        return X_train, X_val, y_train, y_val, prop

    def prepare_train_data(self):
        try:
            data = pd.read_csv(self.path)
            X = data.drop(self.columns_to_drop, 1)
            y = data[self.target]
            prop = (y == 0).sum().astype(float) / (y == 1).sum()
        except Exception as err:
            self.logger.exception(err)
        return X, y, prop

    def validate_prediction(self, model, X_train, y_train, X_val, y_val, scaler):
        try:
            X_test = np.array(scaler.transform(X_val))
            fpr, tpr, thresholds = skl.metrics.roc_curve(y_train, model.predict(X_train))
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            pred = model.predict(X_test)
            pred = (pred > optimal_threshold).astype(int)
            quality_score = skl.metrics.roc_auc_score(y_val, pred)
            self.logger.info(quality_score)
        except Exception as err:
            self.logger.exception(err)
        return quality_score

    def tune_model(self):
        try:
            X_train, X_val, y_train, y_val, prop = self.prepare_tune_data()
            X_train, Transformer = self.fit_transformer(X_train)
            booster = xgb.XGBRegressor(**self.params, scale_pos_weight=1 / prop)
            rs = skl.model_selection.RandomizedSearchCV(cv=self.cv, n_jobs=multiprocessing.cpu_count() // self.n_jobs,
                                                        verbose=100, scoring=self.auc,
                                                        estimator=booster, param_distributions=self.grid,
                                                        n_iter=self.niter, random_state=self.random_state)
            rs.fit(X_train, np.array(y_train))
            result = self.validate_prediction(rs.best_estimator_, X_train, y_train, X_val, y_val)
        except Exception as err:
            self.logger.exception(err)
        return result, rs.best_params_, rs.best_estimator_

    def fit_model(self):
        try:
            X, y, prop = self.prepare_train_data()
            X, Transformer = self.fit_transformer(X)
            booster = xgb.XGBRegressor(**self.params, scale_pos_weight=1 / prop, n_jobs=self.n_jobs)
            booster.fit(X, np.array(y))
        except Exception as err:
            self.logger.exception(err)
        return booster
