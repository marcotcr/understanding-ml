from setuptools import setup

setup(name='lime',
      version='0.1.1.20',
      description='Local Interpretable Model-Agnostic Explanations for machine learning classifiers',
      url='http://github.com/marcotcr/lime',
      author='Marco Tulio Ribeiro/DataScience',
      author_email='marcotcr@gmail.com',
      license='BSD',
      packages=['lime'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn>=0.18'
      ],
      include_package_data=True,
      zip_safe=False)

