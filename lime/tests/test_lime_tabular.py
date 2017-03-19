import unittest

import sys
import os
import numpy as np
import sklearn.datasets
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join("lime"))))
from lime.lime_tabular import LimeTabularExplainer


class TestLimeTabular(unittest.TestCase):
    def test_lime_explainer_bad_regressor(self):
        iris = load_iris()
        train, test, labels_train, labels_test = (
            sklearn.model_selection.train_test_split(iris.data,
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
            exp = explainer.explain_instance(test[i],  # noqa:F841
                                             rf.predict_proba,
                                             num_features=2, top_labels=1,
                                             model_regressor=lasso)

    def test_lime_explainer_good_regressor(self):
        np.random.seed(1)
        iris = load_iris()
        train, test, labels_train, labels_test = (
            sklearn.model_selection.train_test_split(iris.data, iris.target,
                                                      train_size=0.80))

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(train, labels_train)
        i = np.random.randint(0, test.shape[0])

        explainer = LimeTabularExplainer(
            train,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            discretize_continuous=True)

        # to enable more detailed explanation on sampling, enable testing=True for explain_instance
        exp = explainer.explain_instance(test[i], rf.predict_proba,
                                         num_features=2,
                                         model_regressor=LinearRegression(), testing=True)

        self.assertIsNotNone(exp)
        keys = [x[0] for x in exp.as_list()]
        self.assertEquals(1,
                          sum([1 if 'petal width' in x else 0 for x in keys]),
                          "Petal Width is a major feature")
        self.assertEquals(1,
                          sum([1 if 'petal length' in x else 0 for x in keys]),
                          "Petal Length is a major feature")

    def test_lime_explainer_good_regressor_synthetic_data(self):
        X, y = datasets.make_classification(n_samples=1000, n_features=20,
                                            n_informative=2, n_redundant=2, random_state=1)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(X, y)
        instance = np.random.randint(0, X.shape[0])
        print("Index selected for evaluation:{}".format(instance))
        feature_names = ["feature" + str(i) for i in range(20)]
        explainer = LimeTabularExplainer(X,
                                         feature_names=feature_names,
                                         discretize_continuous=True)

        # to enable more detailed explanation on sampling, enable testing=True for explain_instance
        exp = explainer.explain_instance(X[instance], rf.predict_proba, testing=True)

        self.assertIsNotNone(exp)
        self.assertEqual(10, len(exp.as_list()))

    # def test_lime_explainer_no_regressor(self):
    #     np.random.seed(1)
    #     iris = load_iris()
    #     train, test, labels_train, labels_test = (
    #         sklearn.cross_validation.train_test_split(iris.data, iris.target,
    #                                                   train_size=0.80))
    #
    #     rf = RandomForestClassifier(n_estimators=500)
    #     rf.fit(train, labels_train)
    #     i = np.random.randint(0, test.shape[0])
    #
    #     explainer = LimeTabularExplainer(train,
    #                                      feature_names=iris.feature_names,
    #                                      class_names=iris.target_names,
    #                                      discretize_continuous=True)
    #
    #     exp = explainer.explain_instance(test[i], rf.predict_proba,
    #                                      num_features=2)
    #     self.assertIsNotNone(exp)
    #     keys = [x[0] for x in exp.as_list()]
    #     self.assertEquals(1,
    #                       sum([1 if 'petal width' in x else 0 for x in keys]),
    #                       "Petal Width is a major feature")
    #     self.assertEquals(1,
    #                       sum([1 if 'petal length' in x else 0 for x in keys]),
    #                       "Petal Length is a major feature")
    #
    # def test_lime_explainer_entropy_discretizer(self):
    #     np.random.seed(1)
    #     iris = load_iris()
    #     train, test, labels_train, labels_test = (
    #         sklearn.cross_validation.train_test_split(iris.data, iris.target,
    #                                                   train_size=0.80))
    #
    #     rf = RandomForestClassifier(n_estimators=500)
    #     rf.fit(train, labels_train)
    #     i = np.random.randint(0, test.shape[0])
    #
    #     explainer = LimeTabularExplainer(train,
    #                                      feature_names=iris.feature_names,
    #                                      training_labels=labels_train,
    #                                      class_names=iris.target_names,
    #                                      discretize_continuous=True,
    #                                      discretizer='entropy')
    #
    #     exp = explainer.explain_instance(test[i], rf.predict_proba,
    #                                      num_features=2)
    #     self.assertIsNotNone(exp)
    #     keys = [x[0] for x in exp.as_list()]
    #     self.assertEquals(1,
    #                       sum([1 if 'petal width' in x else 0 for x in keys]),
    #                       "Petal Width is a major feature")
    #     self.assertEquals(1,
    #                       sum([1 if 'petal length' in x else 0 for x in keys]),
    #                       "Petal Length is a major feature")


if __name__ == '__main__':
    unittest.main()
