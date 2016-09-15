"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import json
import copy
import numpy as np
import sklearn
import sklearn.preprocessing
from . import lime_base
from . import explanation
from .discretize import QuartileDiscretizer, EntropyDiscretizer


class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""
    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, discretized_feature_names=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
        """
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.scaled_row = scaled_row
        self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             show_table: if False, don't show table visualization.
             show_all: if True, show zero-weighted features in the table.
        """
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]
        out_list = list(zip(self.exp_feature_names, self.feature_values,
                        weights))
        if not show_all:
            out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(out_list), label, div_name)
        return ret


class LimeTabularExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""
    def __init__(self, training_data, labels=None, feature_names=None,
                 categorical_features=None, categorical_names=None,
                 kernel_width=None, verbose=False, class_names=None,
                 feature_selection='auto', discretize_continuous=True):
        """Init function.

        Args:
            training_data: numpy 2d array
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
        """
        self.categorical_names = categorical_names
        self.categorical_features = categorical_features
        if self.categorical_names is None:
            self.categorical_names = {}
        if self.categorical_features is None:
            self.categorical_features = []
        self.discretizer = None
        if discretize_continuous:
            self.discretizer = QuartileDiscretizer(training_data,
                                                   self.categorical_features,
                                                   feature_names,
                                                   labels=labels)
            self.categorical_features = range(training_data.shape[1])
            discretized_training_data = self.discretizer.discretize(
                training_data)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        def kernel(d): return np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose)
        self.scaler = None
        self.class_names = class_names
        self.feature_names = feature_names
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            feature_count = collections.defaultdict(lambda: 0.0)
            column = training_data[:, feature]
            if self.discretizer is not None:
                column = discretized_training_data[:, feature]
                feature_count[0] = 0.
                feature_count[1] = 0.
                feature_count[2] = 0.
                feature_count[3] = 0.
            for value in column:
                feature_count[value] += 1
            values, frequencies = map(list, zip(*(feature_count.items())))
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 sum(frequencies))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    def explain_instance(self, data_row, classifier_fn, labels=(1,),
                         top_labels=None, num_features=10, num_samples=5000,
                         distance_metric='euclidean', model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        data, inverse = self.__data_inverse(data_row, num_samples)
        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

        distances = sklearn.metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        yss = classifier_fn(inverse)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        else:
            self.class_names = list(self.class_names)
        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        def round_stuff(x): return ['%.2f' % a for a in x]
        values = round_stuff(data_row)
        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features
        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                    discretized_instance[f])]

        domain_mapper = TableDomainMapper(
            feature_names, values, scaled_data[0],
            categorical_features=categorical_features,
            discretized_feature_names=discretized_feature_names)
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score) = self.base.explain_instance_with_data(
                scaled_data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        data = np.zeros((num_samples, data_row.shape[0]))
        categorical_features = range(data_row.shape[0])
        if self.discretizer is None:
            data = np.random.normal(
                0, 1, num_samples * data_row.shape[0]).reshape(
                num_samples, data_row.shape[0])
            data = data * self.scaler.scale_ + self.scaler.mean_
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = np.random.choice(values, size=num_samples,
                                              replace=True, p=freqs)
            binary_column = np.array([1 if x == first_row[column]
                                      else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse
