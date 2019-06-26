"""
Discretizers classes, to be used in lime_tabular
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import sklearn
import sklearn.tree
from sklearn.utils import check_random_state


class BaseDiscretizer():
    """
    Abstract class - Build a class that inherits from this class to implement
    a custom discretizer.
    Method bins() is to be redefined in the child class, as it is the actual
    custom part of the discretizer.
    """

    __metaclass__ = ABCMeta  # abstract class

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None,
                 data_stats=None):
        """Initializer
        Args:
            data: numpy 2d array
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. These features will not be discretized.
                Everything else will be considered continuous, and will be
                discretized.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            data_stats: must have 'means', 'stds', 'mins' and 'maxs', use this
                if you don't want these values to be computed from data
        """
        self.to_discretize = ([x for x in range(data.shape[1])
                               if x not in categorical_features])
        self.data_stats = data_stats
        self.names = {}
        self.lambdas = {}
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.random_state = check_random_state(random_state)

        # To override when implementing a custom binning
        bins = self.bins(data, labels)
        bins = [np.unique(x) for x in bins]

        # Read the stats from data_stats if exists
        if data_stats:
            self.means = self.data_stats.get("means")
            self.stds = self.data_stats.get("stds")
            self.mins = self.data_stats.get("mins")
            self.maxs = self.data_stats.get("maxs")

        for feature, qts in zip(self.to_discretize, bins):
            self._update_categorical_names(feature, feature_names, qts)

            def get_index(qts, x):
                result = np.searchsorted(qts, x)
                result = np.where(np.isnan(x), -1, result)

                return result

            self.lambdas[feature] = lambda x, qts=qts: get_index(qts, x)
            # If data stats are provided no need to compute the below set of details
            if data_stats:
                continue

            self._update_feature_dist(data, feature, qts)

            self._set_feature_boundaries(data, feature, qts)

    def _set_feature_boundaries(self, data, feature, qts):
        boundaries = np.min(data[:, feature]), np.max(data[:, feature])
        self.mins[feature] = [boundaries[0]] + qts.tolist()
        self.maxs[feature] = qts.tolist() + [boundaries[1]]

    def _update_feature_dist(self, data, feature, qts):
        discretized = self.lambdas[feature](data[:, feature])
        self.means[feature] = []
        self.stds[feature] = []
        n_bins = qts.shape[0]
        for x in range(-1, n_bins + 1): # -1 corespond for nan
            selection = data[discretized == x, feature]

            mean = 0 if len(selection) == 0 or (x == -1) else np.mean(selection)
            self.means[feature].append(mean)

            std = 0 if len(selection) == 0 or (x == -1) else np.std(selection)
            std += 0.00000000001

            self.stds[feature].append(std)

    def _update_categorical_names(self, feature, feature_names, qts):
        # add nan as one category name
        name = feature_names[feature]
        n_bins = qts.shape[0]  # Actually number of borders (= #bins-1)
        self.names[feature] = ['{} is NaN'.format(name), '%s <= %.2f' % (name, qts[0])]
        for i in range(n_bins - 1):
            self.names[feature].append('%.2f < %s <= %.2f' %
                                       (qts[i], name, qts[i + 1]))
        self.names[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))


    @abstractmethod
    def bins_one(self, data, labels):
        """
        To be overridden
        Returns for each feature to discretize the boundaries
        that form each bin of the discretizer
        """
        raise NotImplementedError("Must override bins() method")

    def discretize(self, data):
        """Discretizes the data.
        Args:
            data: numpy 2d or 1d array
        Returns:
            numpy array of same dimension, discretized.
        """
        ret = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                ret[feature] = int(self.lambdas[feature](ret[feature]))
            else:
                ret[:, feature] = self.lambdas[feature](
                    ret[:, feature]).astype(int)
        return ret

    def undiscretize(self, data):
        ret = data.copy()
        for feature in self.means:
            mins = self.mins[feature]
            maxs = self.maxs[feature]
            means = self.means[feature]
            stds = self.stds[feature]

            def get_inverse(q):
                # return random from normal distribution, bounded by min, max
                if q == -1:
                    result = np.NAN
                else:
                    min_value = min(
                        self.random_state.normal(means[q], stds[q]), maxs[q]
                    )

                    if np.isnan(mins[q]):
                        return min_value
                    else:
                        return max(mins[q], min_value)

                return result

            if len(data.shape) == 1:
                q = int(ret[feature])
                ret[feature] = get_inverse(q)
            else:
                ret[:, feature] = (
                    [get_inverse(int(x)) for x in ret[:, feature]])
        return ret

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            feature_data = data[:, feature]
            selection = ~np.isnan(feature_data)
            not_null_data = feature_data[selection]

            if labels is not None:
                if len(labels) == len(selection):
                    label_at_not_null = labels[selection]
                else:
                    label_at_not_null = labels

                qts = self.bins_one(not_null_data, label_at_not_null)
            else:
                qts = self.bins_one(not_null_data, None)
            bins.append(qts)

        return bins


class StatsDiscretizer(BaseDiscretizer):
    """
        Class to be used to supply the data stats info when discretize_continuous is true
    """

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None,
                 data_stats=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state,
                                 data_stats=data_stats)

    def bins(self, data, labels):
        bins_from_stats = self.data_stats.get("bins")
        bins = []
        if bins_from_stats is not None:
            for feature in self.to_discretize:
                bins_from_stats_feature = bins_from_stats.get(feature)
                if bins_from_stats_feature is not None:
                    qts = np.array(bins_from_stats_feature)
                    bins.append(qts)
        return bins


class QuartileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins_one(self, feature, label):
        qts = np.array(np.percentile(feature, [25, 50, 75]))

        return qts


class DecileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature],
                                         [10, 20, 30, 40, 50, 60, 70, 80, 90]))
            bins.append(qts)
        return bins


class EntropyDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if(labels is None):
            raise ValueError('Labels must be not None when using \
                             EntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 8 bins so max_depth=3
            dt = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=3,
                                                     random_state=self.random_state)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, labels)
            qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if qts.shape[0] == 0:
                qts = np.array([np.median(data[:, feature])])
            else:
                qts = np.sort(qts)

            bins.append(qts)

        return bins
