from setuptools import setup

setup(name='lime',
      version='0.1.1.13',
      description='Local Interpretable Model-Agnostic Explanations for machine learning classifiers',
      url='http://github.com/marcotcr/lime',
      author='Marco Tulio Ribeiro',
      author_email='marcotcr@gmail.com',
      license='BSD',
      packages=['lime'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn'
      ],
      include_package_data=True,
      zip_safe=False)

