from sklearn.svm import SVR
import numpy as np
from sklearn import ensemble


class SVRModel:
    def __init__(self, pre_trained_model=None):
        self.clf = pre_trained_model

    def train(self, X, y):
        self.clf = SVR(kernel='rbf', C=1, gamma=0.1, epsilon=.1)
        self.clf.fit(X, y)

    def predict(self, X):
        if self.clf is None:
            print("No Model exists")
        return self.clf.predict(X)

    def get_model(self):
        return self.clf


class GBModel:
    def __init__(self, pre_trained_model=None):
        self.clf = pre_trained_model

    def train(self, X, y, params):
        self.clf = ensemble.GradientBoostingRegressor(**params)
        self.clf.fit(X, y)

    def predict(self, X):
        if self.clf is None:
            print("No Model exists")
        return self.clf.predict(X)

    def get_model(self):
        return self.clf
