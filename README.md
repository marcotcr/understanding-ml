#Note: We will be removing Lime Text from this module

# lime
This project is about explaining what machine learning classifiers (or models) are doing.
At the moment, we support explaining individual predictions for text classifiers or classifiers that act on tables (numpy arrays of numerical or categorical data), with a package called lime (short for local interpretable model-agnostic explanations).
Lime is based on the work presented in [this paper](http://arxiv.org/abs/1602.04938). Here is a link to the promo video:

<a href="https://www.youtube.com/watch?v=hUnRCxnydCc" target="_blank"><img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/video_screenshot.png" width="450" alt="KDD promo video"/></a>

Our plan is to add more packages that help users understand and interact meaningfully with machine learning.

Lime is able to explain any black box text classifier, with two or more classes. All we require is that the classifier implements a function that takes in raw text or a numpy array and outputs a probability for each class. Support for scikit-learn classifiers is built-in.
## Installation
The lime package is on pypi. Simply run:

    pip install lime
Or clone the repository and run:

    python setup.py install

## Screenshots
Below are some screenshots of lime explanations. These are generated in html, and can be easily produced and embedded in ipython notebooks. We also support visualizations using matplotlib, although they don't look as nice as these ones.
#### Two class case, text
Negative (blue) words indicate atheism, while positive (orange) words indicate christian. The way to interpret the weights by applying them to the prediction probabilities. For example, if we remove the words Host and NNTP from the document, we expect the classifier to predict atheism with probability 0.57 - 0.14 - 0.12 = 0.31.
<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/twoclass.png" />
#### Multiclass case
<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/multiclass.png" />
#### Tabular data
<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/tabular.png" />

## Tutorials and API
For example usage for text classifiers, take a look at the following two tutorials (generated from ipython notebooks):
- [Basic usage, two class. We explain random forest classifiers.](http://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html)
- [Multiclass case](http://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html)

For classifiers that use numerical or categorical data, take a look at the following tutorial (this is newer, so please let me know if you find something wrong):
- [Tabular data](http://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html)

The api reference is available [here](http://lime-ml.readthedocs.io/en/latest/).

## What are explanations?

Intuitively, an explanation is a local linear approximation of the model's behaviour.
While the model may be very complex globally, it is easier to approximate it around the vicinity of a particular instance.
While treating the model as a black box, we perturb the instance we want to explain and learn a sparse linear model around it, as an explanation.
The figure below illustrates the intuition for this procedure. The model's decision function is represented by the blue/pink background, and is clearly nonlinear.
The bright red cross is the instance being explained (let's call it X).
We sample instances around X, and weight them according to their proximity to X (weight here is indicated by size).
We then learn a linear model (dashed line) that approximates the model well in the vicinity of X, but not necessarily globally. For more information, [read our paper](http://arxiv.org/abs/1602.04938), or take a look at [this blog post](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime).

<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/lime.png" width=300px />

## Feedback

If you've tried LIME and are willing to share what went well, and what didn't, please let me know by creating an issue or sending me an email at my github username @gmail.com. I am very interested in understanding when LIME is useful, when it is not, and why.

## Contributing

Please read [this](https://github.com/marcotcr/lime/blob/master/CONTRIBUTING.md).
