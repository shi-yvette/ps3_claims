import hashlib

import numpy as np
import pandas as pd

# The hash of each ID is mapped to one of this many buckets, so the split
# has a resolution of one percentage point.
N_BUCKETS = 100


def _hash_to_bucket(key: str) -> int:
    """Map a string to a bucket between 0 and N_BUCKETS - 1.

    Parameters
    ----------
    key : str
        Value to hash.

    Returns
    -------
    int
        Bucket number. The same key always gives the same bucket.
    """
    # hashlib is stable across sessions and machines. The built-in hash() is
    # salted per Python session for strings, so it would reshuffle the split.
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % N_BUCKETS


def create_sample_split(
    df: pd.DataFrame, id_column: str | list[str], training_frac: float = 0.8
) -> pd.DataFrame:
    """Create sample split based on ID column.

    Each ID is hashed into one of 100 buckets. Rows in the first
    ``training_frac`` share of the buckets are labelled "train" and the rest
    "test", so an ID always lands in the same sample, whatever the other rows
    or their order.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str or list of str
        Name of ID column, or several columns that together identify a row
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Copy of the data with sample column containing train/test split based
        on IDs.

    Raises
    ------
    ValueError
        If ``training_frac`` is not strictly between 0 and 1.
    """
    if not 0 < training_frac < 1:
        raise ValueError(
            f"training_frac must be between 0 and 1 (exclusive), got {training_frac}."
        )

    df = df.copy()
    id_columns = [id_column] if isinstance(id_column, str) else list(id_column)

    # Hash the text form of the ID so that integer and string IDs are treated
    # alike; several columns are joined into one key first.
    keys = df[id_columns[0]].astype(str)
    for column in id_columns[1:]:
        keys = keys + "_" + df[column].astype(str)
    buckets = keys.map(_hash_to_bucket)

    df["sample"] = np.where(buckets / N_BUCKETS < training_frac, "train", "test")
    return df
