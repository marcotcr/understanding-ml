"""
Contains abstract functionality for learning locally linear sparse model.
"""
from __future__ import print_function
import numpy as np
from sklearn import linear_model

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False):
        """Init function

        Args:
            kernel_fn : function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.explanation_model = None
    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the regularization
            parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = linear_model.lars_path(x_vector, weighted_labels,
                                                  method='lasso',
                                                  verbose=False)
        return alphas, coefs
    @staticmethod
    def forward_selection(data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = linear_model.Ridge(alpha=0, fit_intercept=True)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels, sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = linear_model.Ridge(alpha=0, fit_intercept=True)
            clf.fit(data, labels, sample_weight=weights)
            feature_weights = sorted(zip(range(data.shape[0]),
                                         clf.coef_ * data[0]),
                                     key=lambda x: np.abs(x[1]),
                                     reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = (labels - np.average(labels, weights=weights)) * np.sqrt(weights)
            used_features = range(weighted_data.shape[1])
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)


    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto', give_score=False):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model. This
                                     is costly when num_features is high
                'highest_weights': selects the features that have the highest
                                   product of absolute weight * original data
                                   point when learning with all the
                                   features
                'lasso_path': chooses features based on the lasso regularization
                              path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                        'highest_weights' otherwise.

        Returns:
            (intercept, exp):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
        """
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)

        self.explanation_model = linear_model.Ridge(alpha=1, fit_intercept=True)
        self.explanation_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        if self.verbose:
            local_pred = self.explanation_model.predict(
                neighborhood_data[0, used_features].reshape(1, -1))
            print('Intercept', self.explanation_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return self.explanation_model.intercept_, sorted(zip(used_features, self.explanation_model.coef_),
                      key=lambda x: np.abs(x[1]), reverse=True)
