import numpy as np
import numpy.typing as npt
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import auc


def evaluate_predictions(
    predictions: npt.ArrayLike,
    actuals: npt.ArrayLike,
    sample_weight: npt.ArrayLike,
    tweedie_power: float = 1.5,
) -> pd.DataFrame:
    """Compute performance metrics for a set of predictions.

    The sample weight enters every metric. For the claims data it is the
    exposure and the predictions and actuals are pure premiums.

    - ``bias``: weighted mean prediction minus weighted mean actual, divided
      by the weighted mean actual. A value of -0.05 means the predictions are
      5% too low on average.
    - ``deviance``: weighted mean Tweedie deviance.
    - ``mae``: weighted mean absolute error.
    - ``rmse``: square root of the weighted mean squared error.
    - ``gini``: one minus twice the area under the Lorenz curve, as in
      ``analyses/ps3_script.py``. Observations are ordered by increasing
      prediction, each counts equally on the horizontal axis, and the curve
      cumulates actual times weight, which is the claim amount.

    Parameters
    ----------
    predictions : array-like of shape (n_samples,)
        Predicted values. They must be positive for the deviance.
    actuals : array-like of shape (n_samples,)
        Observed values.
    sample_weight : array-like of shape (n_samples,)
        Weight of each observation.
    tweedie_power : float, optional
        Power of the Tweedie deviance, by default 1.5

    Returns
    -------
    pd.DataFrame
        One column ``value`` with the metric names as index.
    """
    y_pred = np.asarray(predictions, dtype=float)
    y_true = np.asarray(actuals, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)

    mean_actual = np.average(y_true, weights=weight)
    bias = (np.average(y_pred, weights=weight) - mean_actual) / mean_actual
    deviance = TweedieDistribution(tweedie_power).deviance(
        y_true, y_pred, sample_weight=weight
    ) / np.sum(weight)
    mae = np.average(np.abs(y_true - y_pred), weights=weight)
    rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=weight))

    ranking = np.argsort(y_pred)
    cumulated_actuals = np.cumsum(y_true[ranking] * weight[ranking])
    cumulated_actuals /= cumulated_actuals[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_actuals))
    gini = 1 - 2 * auc(cumulated_samples, cumulated_actuals)

    return pd.DataFrame(
        {"value": [bias, deviance, mae, rmse, gini]},
        index=["bias", "deviance", "mae", "rmse", "gini"],
    )
