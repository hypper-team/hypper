![](logo/logo1.png)

![PyPI](https://img.shields.io/pypi/v/hypper) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hypper) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/hypper) ![build](https://github.com/hypper-team/hypper/actions/workflows/main.yml/badge.svg) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/hypper?period=total&units=none&left_color=grey&right_color=yellowgreen&left_text=downloads)](https://pepy.tech/project/hypper) ![PyPI - License](https://img.shields.io/pypi/l/hypper)

Hypper is a data-mining Python library for binary classification. It uses hypergraph-based methods to explore datasets for the purpose of undersampling, feature selection and binary classification.

Hypper provides an easy-to-use interface familiar to well-recognized Scikit-Learn API. 

The primary goal of this library is to provide a tool for handling datasets consisting of mainly categorical features. Novel hypergraph-based methods proposed in the Hypper library were benchmarked against the alternative solutions and achieved satisfactory results. More details can be found in scientific papers presented in the section below.

## Installation
```bash
pip install hypper
```
Local installations
``` bash
pip install -e .['documentation'] # documentation
pip install -e .['develop'] # development (with testing)
pip install -e .['benchmarking'] # benchmarking scripts
pip install -e .['all'] # install everything
```

## Tutorials:
[![](https://colab.research.google.com/assets/colab-badge.svg)  1. Introduction to data mining with Hypper](https://colab.research.google.com/drive/1JntX8z3-e0qhCSjxpjYnPmfR2Iy09e15?usp=sharing)

## Testing
```bash
pytest
```
## Important links
* Source code - [https://github.com/hypper-team/hypper](https://github.com/hypper-team/hypper)
* Documentation - [https://hypper-team.github.io/hypper.html](https://hypper-team.github.io/hypper.html)

## Citation
```
@ARTICLE{Misiorek2022-ru,
  title     = "Hypergraph-based importance assessment for binary classification
               data",
  author    = "Misiorek, Pawel and Janowski, Szymon",
  abstract  = "AbstractWe present a novel hypergraph-based framework enabling
               an assessment of the importance of binary classification data
               elements. Specifically, we apply the hypergraph model to rate
               data samples' and categorical feature values' relevance to
               classification labels. The proposed Hypergraph-based Importance
               ratings are theoretically grounded on the hypergraph cut
               conductance minimization concept. As a result of using
               hypergraph representation, which is a lossless representation
               from the perspective of higher-order relationships in data, our
               approach allows for more precise exploitation of the information
               on feature and sample coincidences. The solution was tested
               using two scenarios: undersampling for imbalanced classification
               data and feature selection. The experimentation results have
               proven the good quality of the new approach when compared with
               other state-of-the-art and baseline methods for both scenarios
               measured using the average precision evaluation metric.",
  journal   = "Knowl. Inf. Syst.",
  publisher = "Springer Science and Business Media LLC",
  month     =  dec,
  year      =  2022,
  copyright = "https://creativecommons.org/licenses/by/4.0",
  language  = "en"
}
```
