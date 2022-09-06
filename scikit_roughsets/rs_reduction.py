import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from scikit_roughsets.roughsets import RoughSetsReducer


class RoughSetsSelector(BaseEstimator, SelectorMixin):

    def _get_support_mask(self):
        return self.mask_

    def fit(self, X, y=None):
        # Missing values are not supported yet!
        if np.isnan(X).any():
            raise ValueError("X must not contain any missing values")
        if np.isnan(y).any():
            raise ValueError("y must not contain any missing values")
        # Check that X and Y contains only integer values
        if not np.all(np.equal(np.mod(X, 1), 0)):
            raise ValueError("X must contain only integer values")
        if not np.all(np.equal(np.mod(y, 1), 0)):
            raise ValueError("y must contain only integer values")

        reducer = RoughSetsReducer()
        selected_ = reducer.reduce(X, y)
        B_unique_sorted, B_idx = np.unique(np.array(range(X.shape[1])), return_index=True)
        B_unique_sorted = B_unique_sorted + 1  # Shift elements by one, as RS index array starts by one
        self.mask_ = np.in1d(B_unique_sorted, selected_, assume_unique=True)

        if self.mask_.size == 0:
            raise ValueError("No features were selected by rough sets reducer")
        return self
