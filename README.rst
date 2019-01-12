scikit-roughsets
================

[![Build Status](https://travis-ci.org/paudan/scikit-roughsets.svg?branch=master)

This is an implementation of rough sets feature reduction algorithm, based on MATLAB code from
`Dingyu Xue, YangQuan Chen. Solving applied mathematical problems with MATLAB <https://books.google.lt/books?id=V4vulPEc29kC>`_. Integration with *scikit-learn* package is also provided.


Installation
------------

The package can be easily installed using Python's *pip* utility.

Usage
-----

The usage is very straightforward, identical to scikit's feature selection module:

.. code:: python

    from scikit_roughsets.rs_reduction import RoughSetsSelector
    import numpy as np

    y = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T
    X = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                  [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                  [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                  [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                  [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                  [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                  [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1]])

    selector = RoughSetsSelector()
    X_selected = selector.fit(X, y).transform(X)

Several restrictions apply to its current use:

- *X* must be an integer matrix, and *y* must must be an integer array
- It does not work with NaN values, thus, initial preprocessing must be performed by the user
