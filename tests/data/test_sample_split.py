import numpy as np
import pandas as pd
import pytest

from ps3.data import create_sample_split

N_ROWS = 10_000
SEED = 42


@pytest.fixture
def df() -> pd.DataFrame:
    """Data with a unique integer ID, like IDpol in the claims data."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {"id": np.arange(1, N_ROWS + 1), "value": rng.normal(size=N_ROWS)}
    )


def test_adds_sample_column_with_train_and_test(df: pd.DataFrame) -> None:
    result = create_sample_split(df, id_column="id")

    assert list(result.columns) == ["id", "value", "sample"]
    assert set(result["sample"]) == {"train", "test"}
    pd.testing.assert_frame_equal(result[["id", "value"]], df)


def test_does_not_modify_input(df: pd.DataFrame) -> None:
    original = df.copy()

    create_sample_split(df, id_column="id")

    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("training_frac", [0.5, 0.8, 0.9])
def test_train_share_is_close_to_training_frac(
    df: pd.DataFrame, training_frac: float
) -> None:
    result = create_sample_split(df, id_column="id", training_frac=training_frac)

    train_share = (result["sample"] == "train").mean()
    assert train_share == pytest.approx(training_frac, abs=0.02)


def test_default_training_frac_is_80_percent(df: pd.DataFrame) -> None:
    default = create_sample_split(df, id_column="id")
    explicit = create_sample_split(df, id_column="id", training_frac=0.8)

    pd.testing.assert_frame_equal(default, explicit)


def test_same_id_always_lands_in_same_sample(df: pd.DataFrame) -> None:
    """The split of an ID depends on neither row order nor the other rows."""
    full = create_sample_split(df, id_column="id")
    shuffled_subset = df.sample(frac=0.3, random_state=SEED)
    subset = create_sample_split(shuffled_subset, id_column="id")

    expected = full.set_index("id")["sample"].loc[subset["id"]]
    assert (subset["sample"].to_numpy() == expected.to_numpy()).all()


def test_split_is_fixed_across_sessions() -> None:
    """Guard against a hash that changes between runs, such as hash()."""
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    result = create_sample_split(df, id_column="id", training_frac=0.5)

    # Computed once with the md5-based split and written down here.
    expected = [
        "train", "train", "test", "train", "test",
        "train", "test", "train", "test", "train",
    ]  # fmt: skip
    assert result["sample"].tolist() == expected


def test_integer_and_string_ids_give_same_split(df: pd.DataFrame) -> None:
    as_int = create_sample_split(df, id_column="id")
    as_str = create_sample_split(df.astype({"id": str}), id_column="id")

    assert as_int["sample"].tolist() == as_str["sample"].tolist()


def test_split_on_several_columns() -> None:
    df = pd.DataFrame(
        {
            "policy": np.repeat(np.arange(1_000), 5),
            "year": np.tile(np.arange(2015, 2020), 1_000),
        }
    )

    result = create_sample_split(df, id_column=["policy", "year"])
    reordered = create_sample_split(df.iloc[::-1], id_column=["policy", "year"])

    assert (result["sample"] == "train").mean() == pytest.approx(0.8, abs=0.02)
    pd.testing.assert_series_equal(result["sample"], reordered["sample"].iloc[::-1])


@pytest.mark.parametrize("training_frac", [-0.1, 0, 1, 1.5])
def test_invalid_training_frac_raises(df: pd.DataFrame, training_frac: float) -> None:
    with pytest.raises(ValueError, match="training_frac must be between 0 and 1"):
        create_sample_split(df, id_column="id", training_frac=training_frac)
