import unittest

import numpy as np
import sklearn.datasets
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from lime.discretizer.discretize import EntropyDiscretizer
from lime.lime_tabular import LimeTabularExplainer


class TestLimeTabular(unittest.TestCase):
    def test_lime_explainer_bad_regressor(self):
        iris = load_iris()
        train, test, labels_train, labels_test = (
            sklearn.cross_validation.train_test_split(iris.data,
                                                      iris.target,
                                                      train_size=0.80))

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(train, labels_train)
        lasso = Lasso(alpha=1, fit_intercept=True)
        i = np.random.randint(0, test.shape[0])
        with self.assertRaises(TypeError):
            explainer = LimeTabularExplainer(
                train,
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                discretize_continuous=True)

            exp = explainer.explain_instance(test[i], rf.predict_proba, # noqa:F841
                                             num_features=2, top_labels=1,
                                             model_regressor=lasso)

    def test_lime_explainer_good_regressor(self):
        np.random.seed(1)
        iris = load_iris()
        train, test, labels_train, labels_test = (
            sklearn.cross_validation.train_test_split(iris.data, iris.target,
                                                      train_size=0.80))

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(train, labels_train)
        i = np.random.randint(0, test.shape[0])

        explainer = LimeTabularExplainer(
            train,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            discretize_continuous=True)

        exp = explainer.explain_instance(test[i], rf.predict_proba,
                                         num_features=2,
                                         model_regressor=LinearRegression())
        self.assertIsNotNone(exp)

    def test_lime_explainer_no_regressor(self):
        np.random.seed(1)
        iris = load_iris()
        train, test, labels_train, labels_test = (
            sklearn.cross_validation.train_test_split(iris.data, iris.target,
                                                      train_size=0.80))

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(train, labels_train)
        i = np.random.randint(0, test.shape[0])

        explainer = LimeTabularExplainer(train,
                                         feature_names=iris.feature_names,
                                         class_names=iris.target_names,
                                         discretize_continuous=True)

        exp = explainer.explain_instance(test[i], rf.predict_proba,
                                         num_features=2)
        self.assertIsNotNone(exp)

    def test_lime_explainer_entropy_discretization(self):
        np.random.seed(1)
        iris = load_iris()
        train, test, labels_train, labels_test = (
            sklearn.cross_validation.train_test_split(iris.data, iris.target,
                                                      train_size=0.80))

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(train, labels_train)
        i = np.random.randint(0, test.shape[0])

        explainer = LimeTabularExplainer(train, labels=labels_train,
                                         feature_names=iris.feature_names,
                                         class_names=iris.target_names,
                                         discretize_continuous=True, discretizer=EntropyDiscretizer)

        exp = explainer.explain_instance(test[i], rf.predict_proba,
                                         num_features=2)
        self.assertIsNotNone(exp)

if __name__ == '__main__':
        unittest.main()
