from setuptools import setup, find_packages
import sys
python_version = sys.version_info
pillow_version = 'pillow==7.0.0' if python_version[1] is 8 else 'pillow==5.4.1'

setup(name='lime',
      version='0.2.0.0',
      description='Local Interpretable Model-Agnostic Explanations for machine learning classifiers',
      url='http://github.com/marcotcr/lime',
      author='Marco Tulio Ribeiro',
      author_email='marcotcr@gmail.com',
      license='BSD',
      packages=find_packages(exclude=['js', 'node_modules', 'tests']),
      python_requires='>=3.5',
      install_requires=[
          'matplotlib',
          'numpy',
          'scipy',
          'tqdm',
          pillow_version,
          'scikit-learn>=0.18',
          'scikit-image>=0.12',
      ],
      extras_require={
          'dev': ['pytest', 'flake8'],
      },
      include_package_data=True,
      zip_safe=False)
