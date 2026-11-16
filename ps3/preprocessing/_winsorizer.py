from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric features at quantiles learned from the training data.

    Values below the lower quantile are set to the lower quantile and values
    above the upper quantile are set to the upper quantile. For 2D input the
    quantiles are computed for each column separately.

    Parameters
    ----------
    lower_quantile : float, optional
        Quantile below which values are clipped, by default 0.05
    upper_quantile : float, optional
        Quantile above which values are clipped, by default 0.95

    Attributes
    ----------
    lower_quantile_ : float or np.ndarray
        Lower clipping bound(s) computed in ``fit``, one per column.
    upper_quantile_ : float or np.ndarray
        Upper clipping bound(s) computed in ``fit``, one per column.
    """

    def __init__(
        self, lower_quantile: float = 0.05, upper_quantile: float = 0.95
    ) -> None:
        # Only store the arguments here, so that get_params, set_params and
        # clone work. Validation happens in fit.
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: Any, y: Any = None) -> "Winsorizer":
        """Compute the clipping bounds from X.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Data used to compute the quantiles.
        y : None
            Ignored. Present for compatibility with the scikit-learn API.

        Returns
        -------
        Winsorizer
            The fitted transformer.

        Raises
        ------
        ValueError
            If the quantiles are not ordered as
            0 <= lower_quantile <= upper_quantile <= 1.
        """
        if not 0 <= self.lower_quantile <= self.upper_quantile <= 1:
            raise ValueError(
                "Quantiles must satisfy 0 <= lower_quantile <= upper_quantile <= 1, "
                f"got lower_quantile={self.lower_quantile} and "
                f"upper_quantile={self.upper_quantile}."
            )

        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X: Any) -> Any:
        """Clip X at the bounds computed in ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Data to clip.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Clipped copy of X, of the same type and shape as X.
        """
        check_is_fitted(self)
        return np.clip(X, self.lower_quantile_, self.upper_quantile_)
