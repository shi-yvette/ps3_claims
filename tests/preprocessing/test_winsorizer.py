import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ps3.preprocessing import Winsorizer

SEED = 42


@pytest.fixture
def X() -> np.ndarray:
    """Standard normal sample drawn with a fixed seed."""
    rng = np.random.default_rng(SEED)
    return rng.normal(0, 1, 1000)


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(
    X: np.ndarray, lower_quantile: float, upper_quantile: float
) -> None:
    expected_lower = np.quantile(X, lower_quantile)
    expected_upper = np.quantile(X, upper_quantile)

    winsorizer = Winsorizer(lower_quantile, upper_quantile)
    Xt = winsorizer.fit_transform(X)

    assert winsorizer.lower_quantile_ == pytest.approx(expected_lower)
    assert winsorizer.upper_quantile_ == pytest.approx(expected_upper)
    assert Xt.shape == X.shape
    assert Xt.min() == pytest.approx(expected_lower)
    assert Xt.max() == pytest.approx(expected_upper)
    # Values between the bounds must stay as they are.
    inside = (X >= expected_lower) & (X <= expected_upper)
    np.testing.assert_array_equal(Xt[inside], X[inside])


def test_transform_uses_bounds_learned_in_fit() -> None:
    """New data is clipped at the training quantiles, not at its own."""
    X_train = np.arange(101, dtype=float)  # quantiles are easy: q -> 100 * q
    X_test = np.array([-50.0, 0.0, 50.0, 100.0, 500.0])

    winsorizer = Winsorizer(lower_quantile=0.1, upper_quantile=0.9).fit(X_train)
    Xt = winsorizer.transform(X_test)

    assert winsorizer.lower_quantile_ == pytest.approx(10.0)
    assert winsorizer.upper_quantile_ == pytest.approx(90.0)
    np.testing.assert_allclose(Xt, [10.0, 10.0, 50.0, 90.0, 90.0])


def test_columns_are_clipped_separately() -> None:
    rng = np.random.default_rng(SEED)
    X = np.column_stack([rng.normal(0, 1, 1000), rng.normal(100, 10, 1000)])

    winsorizer = Winsorizer(lower_quantile=0.05, upper_quantile=0.95)
    Xt = winsorizer.fit_transform(X)

    np.testing.assert_allclose(winsorizer.lower_quantile_, np.quantile(X, 0.05, axis=0))
    np.testing.assert_allclose(winsorizer.upper_quantile_, np.quantile(X, 0.95, axis=0))
    np.testing.assert_allclose(Xt.min(axis=0), winsorizer.lower_quantile_)
    np.testing.assert_allclose(Xt.max(axis=0), winsorizer.upper_quantile_)


def test_dataframe_input_keeps_columns_and_index() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(
        {"a": rng.normal(0, 1, 100), "b": rng.normal(100, 10, 100)},
        index=np.arange(100, 200),
    )

    Xt = Winsorizer(lower_quantile=0.1, upper_quantile=0.9).fit_transform(df)

    assert isinstance(Xt, pd.DataFrame)
    pd.testing.assert_index_equal(Xt.index, df.index)
    pd.testing.assert_frame_equal(
        Xt, df.clip(df.quantile(0.1), df.quantile(0.9), axis=1)
    )


def test_transform_does_not_modify_input(X: np.ndarray) -> None:
    original = X.copy()

    Winsorizer(lower_quantile=0.05, upper_quantile=0.95).fit_transform(X)

    np.testing.assert_array_equal(X, original)


def test_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        Winsorizer().transform(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile",
    [(-0.1, 0.9), (0.1, 1.1), (0.9, 0.1)],
)
def test_invalid_quantiles_raise(lower_quantile: float, upper_quantile: float) -> None:
    winsorizer = Winsorizer(lower_quantile, upper_quantile)

    with pytest.raises(ValueError, match="Quantiles must satisfy"):
        winsorizer.fit(np.array([1.0, 2.0, 3.0]))


def test_works_as_a_pipeline_step() -> None:
    """In a Pipeline, the test data is clipped at the training bounds."""
    X_train = np.arange(101, dtype=float).reshape(-1, 1)
    X_test = np.array([[-50.0], [50.0], [500.0]])
    pipeline = Pipeline(
        [
            ("winsorize", Winsorizer(lower_quantile=0.1, upper_quantile=0.9)),
            ("scale", StandardScaler()),
        ]
    )

    pipeline.fit(X_train)
    Xt = pipeline.transform(X_test)

    scaler = pipeline["scale"]
    expected = (np.array([[10.0], [50.0], [90.0]]) - scaler.mean_) / scaler.scale_
    np.testing.assert_allclose(Xt, expected)


def test_follows_sklearn_parameter_conventions() -> None:
    winsorizer = Winsorizer(lower_quantile=0.1, upper_quantile=0.8)

    assert winsorizer.get_params() == {"lower_quantile": 0.1, "upper_quantile": 0.8}
    assert clone(winsorizer).get_params() == winsorizer.get_params()
    assert winsorizer.fit(np.arange(10.0)) is winsorizer
