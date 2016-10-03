import unittest

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from lime.lime_text import LimeTextExplainer


class TestLimeText(unittest.TestCase):
    def test_lime_text_explainer_good_regressor(self):
        from sklearn.datasets import fetch_20newsgroups
        categories = ['alt.atheism', 'soc.religion.christian']
        newsgroups_train = fetch_20newsgroups(subset='train',
                                              categories=categories)
        newsgroups_test = fetch_20newsgroups(subset='test',
                                             categories=categories)
        class_names = ['atheism', 'christian']
        vectorizer = TfidfVectorizer(lowercase=False)
        train_vectors = vectorizer.fit_transform(newsgroups_train.data)
        test_vectors = vectorizer.transform(newsgroups_test.data)
        nb = MultinomialNB(alpha=.01)
        nb.fit(train_vectors, newsgroups_train.target)
        pred = nb.predict(test_vectors)
        f1_score(newsgroups_test.target, pred, average='weighted')
        c = make_pipeline(vectorizer, nb)
        explainer = LimeTextExplainer(class_names=class_names)
        idx = 83
        exp = explainer.explain_instance(newsgroups_test.data[idx],
                                         c.predict_proba, num_features=6)
        self.assertIsNotNone(exp)
        self.assertEquals(6, len(exp.as_list()))

    def test_lime_text_explainer_bad_regressor(self):
        from sklearn.datasets import fetch_20newsgroups
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')
        # making class names shorter
        class_names = [x.split('.')[-1] if 'misc' not in x
                       else '.'.join(x.split('.')[-2:])
                       for x in newsgroups_train.target_names]
        class_names[3] = 'pc.hardware'
        class_names[4] = 'mac.hardware'
        vectorizer = TfidfVectorizer(lowercase=False)
        train_vectors = vectorizer.fit_transform(newsgroups_train.data)
        test_vectors = vectorizer.transform(newsgroups_test.data)
        nb = MultinomialNB(alpha=.01)
        nb.fit(train_vectors, newsgroups_train.target)
        pred = nb.predict(test_vectors)
        f1_score(newsgroups_test.target, pred, average='weighted')
        c = make_pipeline(vectorizer, nb)
        explainer = LimeTextExplainer(class_names=class_names)
        idx = 1340
        with self.assertRaises(TypeError):
            exp = explainer.explain_instance(# noqa:F841
                newsgroups_test.data[idx], c.predict_proba, num_features=6,
                labels=[0, 17], model_regressor=Lasso())


if __name__ == '__main__':
    unittest.main()
