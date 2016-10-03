"""
Explanation class, with visualization functions.
"""
from __future__ import unicode_literals
from io import open
import os
import os.path
import json
import string
import numpy as np


def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))


class DomainMapper(object):
    """Class for mapping features to the specific domain.

    The idea is that there would be a subclass for each domain (text, tables,
    images, etc), so that we can have a general Explanation class, and separate
    out the specifics of visualizing features in here.
    """
    def __init__(self):
        pass

    def map_exp_ids(self, exp, **kwargs):
        """Maps the feature ids to concrete names.

        Default behaviour is the identity function. Subclasses can implement
        this as they see fit.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            kwargs: optional keyword arguments

        Returns:
            exp: list of tuples [(name, weight), (name, weight)...]
        """
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name,
                                **kwargs):
        """Produces html for visualizing the instance.

        Default behaviour does nothing. Subclasses can implement this as they
        see fit.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             kwargs: optional keyword arguments

        Returns:
             js code for visualizing the instance
        """
        return ''


class Explanation(object):
    """Object returned by explainers."""
    def __init__(self, domain_mapper, class_names=None):
        """Initializer.

        Args:
            domain_mapper: must inherit from DomainMapper class
            class_names: list of class names
        """
        self.domain_mapper = domain_mapper
        self.class_names = class_names
        self.local_exp = {}
        self.intercept = {}
        self.top_labels = None
        self.predict_proba = None
        self.score = None

    def available_labels(self):
        """Returns the list of labels for which we have any explanations."""
        if self.top_labels:
            return list(self.top_labels)
        return list(self.local_exp.keys())

    def as_list(self, label=1, **kwargs):
        """Returns the explanation as a list.

        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            list of tuples (representation, weight), where representation is
            given by domain_mapper. Weight is a float.
        """
        return self.domain_mapper.map_exp_ids(self.local_exp[label], **kwargs)

    def as_map(self):
        """Returns the map of explanations.

        Returns:
            Map from label to list of tuples (feature_id, weight).
        """
        return self.local_exp

    def as_pyplot_figure(self, label=1, **kwargs):
        """Returns the explanation as a pyplot figure.

        Will throw an error if you don't have matplotlib installed
        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            pyplot figure (barchart).
        """
        import matplotlib.pyplot as plt
        exp = self.as_list(label, **kwargs)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        plt.title('Local explanation for class %s' % self.class_names[label])
        return fig

    def show_in_notebook(self, labels=None, predict_proba=True, **kwargs):
        """Shows html explanation in ipython notebook.

           See as_html for parameters.
           This will throw an error if you don't have IPython installed"""
        from IPython.core.display import display, HTML
        display(HTML(self.as_html(labels, predict_proba, **kwargs)))

    def save_to_file(self, file_path, labels=None, predict_proba=True,
                     **kwargs):
        """Saves html explanation to file. See as_html for paramaters.

        Params:
            file_path: file to save explanations to
        """
        file_ = open(file_path, 'w')
        file_.write(self.as_html(labels, predict_proba, **kwargs))
        file_.close()

    def as_html(self, labels=None, predict_proba=True, **kwargs):
        """Returns the explanation as an html page.

        Args:
            labels: desired labels to show explanations for (as barcharts).
                If you ask for a label for which an explanation wasn't
                computed, will throw an exception. If None, will show
                explanations for all available labels.
            predict_proba: if true, add  barchart with prediction probabilities
                for the top classes.
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            code for an html page, including javascript includes.
        """
        def jsonize(x): return json.dumps(x)
        if labels is None:
            labels = self.available_labels()
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'),
                      encoding="utf8").read()

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        random_id = id_generator()
        out += u'''
        <div class="lime top_div" id="top_div%s"></div>
        ''' % random_id
        predict_proba_js = ''
        if predict_proba:
            predict_proba_js = u'''
            var pp_div = top_div.append('div')
                                .classed('lime predict_proba', true);
            var pp_svg = pp_div.append('svg').style('width', '100%%');
            var pp = new lime.PredictProba(pp_svg, %s, %s);
            ''' % (jsonize(self.class_names),
                   jsonize(list(self.predict_proba.astype(float))))

        exp_js = '''var exp_div;
            var exp = new lime.Explanation(%s);
        ''' % (jsonize(self.class_names))
        for label in labels:
            exp = jsonize(self.as_list(label))
            exp_js += u'''
            exp_div = top_div.append('div').classed('lime explanation', true);
            exp.show(%s, %d, exp_div);
            ''' % (exp, label)
        raw_js = '''var raw_div = top_div.append('div');'''
        raw_js += self.domain_mapper.visualize_instance_html(
            self.local_exp[labels[0]], labels[0], 'raw_div', 'exp', **kwargs)
        out += u'''
        <script>
        var top_div = d3.select('#top_div%s').classed('lime top_div', true);
        %s
        %s
        %s
        </script>
        ''' % (random_id, predict_proba_js, exp_js, raw_js)
        out += u'</body></html>'
        return out
