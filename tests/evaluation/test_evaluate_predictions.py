import numpy as np
import pytest
from sklearn.metrics import mean_tweedie_deviance

from ps3.evaluation import evaluate_predictions

SEED = 42


def test_returns_one_value_per_metric() -> None:
    result = evaluate_predictions([1.0, 2.0], [1.0, 3.0], [1.0, 1.0])

    assert result.shape == (5, 1)
    assert list(result.index) == ["bias", "deviance", "mae", "rmse", "gini"]
    assert list(result.columns) == ["value"]


def test_perfect_predictions_have_no_error() -> None:
    actuals = [1.0, 2.0, 5.0]

    result = evaluate_predictions(actuals, actuals, [0.5, 1.0, 0.2])["value"]

    assert result[["bias", "deviance", "mae", "rmse"]].to_numpy() == pytest.approx(0)


# Actuals 0, 1 and 4 with a prediction of 1 each. With power 1.5 the unit
# deviances are 4, 0 and 4, the absolute errors 1, 0 and 3.
@pytest.mark.parametrize(
    "sample_weight, expected",
    [
        (
            [1, 1, 1],
            {"bias": -0.4, "deviance": 8 / 3, "mae": 4 / 3, "rmse": np.sqrt(10 / 3)},
        ),
        (
            [1, 1, 2],
            {"bias": -5 / 9, "deviance": 3.0, "mae": 1.75, "rmse": np.sqrt(4.75)},
        ),
    ],
)
def test_metrics_match_hand_computed_values(
    sample_weight: list[int], expected: dict[str, float]
) -> None:
    result = evaluate_predictions([1, 1, 1], [0, 1, 4], sample_weight)["value"]

    for metric, value in expected.items():
        assert result[metric] == pytest.approx(value)


# With three observations the Lorenz curve has the points 0, 0.5 and 1 on the
# horizontal axis, so the area under it is easy to compute by hand.
@pytest.mark.parametrize(
    "predictions, actuals, sample_weight, expected_gini",
    [
        ([1, 2, 3], [0, 0, 6], [1, 1, 1], 0.5),  # largest claim ranked last
        ([3, 2, 1], [0, 0, 6], [1, 1, 1], -1.0),  # largest claim ranked first
        ([1, 2, 3], [0, 2, 2], [1, 1, 2], 1 / 6),  # weights turn 2, 2 into 2, 4
    ],
)
def test_gini_matches_hand_computed_values(
    predictions: list[int],
    actuals: list[int],
    sample_weight: list[int],
    expected_gini: float,
) -> None:
    result = evaluate_predictions(predictions, actuals, sample_weight)["value"]

    assert result["gini"] == pytest.approx(expected_gini)


def test_rescaling_the_weights_changes_nothing() -> None:
    rng = np.random.default_rng(SEED)
    predictions = rng.uniform(1, 10, 100)
    actuals = rng.uniform(0, 10, 100)
    sample_weight = rng.uniform(0.1, 1, 100)

    result = evaluate_predictions(predictions, actuals, sample_weight)
    rescaled = evaluate_predictions(predictions, actuals, 10 * sample_weight)

    np.testing.assert_allclose(rescaled["value"], result["value"])


@pytest.mark.parametrize("tweedie_power", [1.5, 1.9])
def test_deviance_agrees_with_scikit_learn(tweedie_power: float) -> None:
    rng = np.random.default_rng(SEED)
    predictions = rng.uniform(1, 10, 100)
    actuals = rng.uniform(0, 10, 100)
    sample_weight = rng.uniform(0.1, 1, 100)

    result = evaluate_predictions(
        predictions, actuals, sample_weight, tweedie_power=tweedie_power
    )["value"]

    assert result["deviance"] == pytest.approx(
        mean_tweedie_deviance(
            actuals, predictions, sample_weight=sample_weight, power=tweedie_power
        )
    )
