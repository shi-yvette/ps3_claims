from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

# Raw files are kept in the "data" folder at the root of the repository
# (ignored by git), so the 36 MB download only happens on the first run.
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FREQ_URL = "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff"
SEV_URL = "https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff"


def _download_once(url: str, filename: str) -> Path:
    """Download a file into the data folder unless it is already there.

    Parameters
    ----------
    url : str
        Address of the file.
    filename : str
        Name under which the file is stored in the data folder.

    Returns
    -------
    Path
        Location of the local copy.
    """
    path = DATA_DIR / filename
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urlretrieve(url, path)
    return path


def load_transform() -> pd.DataFrame:
    """Load and transform data from OpenML.

    The two raw files are downloaded on the first call and read from the local
    data folder afterwards.

    Source:
    https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html#

    Summary of transformations:

    1. We cut the number of claims to a maximum of 4, as is done in the case study paper
       (Case-study authors suspect a data error. See section 1 of their paper for
       details).
    2. We cut the exposure to a maximum of 1, as is done in the case study paper
       (Case-study authors suspect a data error. See section 1 of their paper for
       details).
    3. We define ``'ClaimAmountCut'`` as the the claim amount cut at 100'000 per
       single claim (before aggregation per policy). Reason: For large claims,
       extreme value theory might apply. 100'000 is the 0.9984 quantile, any claims
       larger account for 25% of the overall claim amount. This is a well known
       phenomenon for third-party liability.
    4. We aggregate the total claim amounts per policy ID and join them to
       ``freMTPL2freq``.
    5. We fix ``'ClaimNb'`` as the claim number with claim amount greater zero.
    6. ``'VehPower'``, ``'VehAge'``, and ``'DrivAge'`` are clipped and/or digitized
       into bins so they can be used as categoricals later on.

    Returns
    -------
    pd.DataFrame
        One row per policy, with the claim counts, exposure, claim amounts and
        the policy features.
    """
    # load the datasets
    # first row (=column names) uses "", all other rows use ''
    # use '' as quotechar as it is easier to change column names
    df = pd.read_csv(_download_once(FREQ_URL, "freMTPL2freq.csv"), quotechar="'")

    # rename column names '"name"' => 'name'
    df = df.rename(lambda x: x.replace('"', ""), axis="columns")
    df["IDpol"] = df["IDpol"].astype(np.int64)
    df.set_index("IDpol", inplace=True)

    df_sev = pd.read_csv(_download_once(SEV_URL, "freMTPL2sev.csv"), index_col=0)

    # join ClaimAmount from df_sev to df:
    #   1. cut ClaimAmount at 100_000
    #   2. aggregate ClaimAmount per IDpol
    #   3. join by IDpol
    df_sev["ClaimAmountCut"] = df_sev["ClaimAmount"].clip(upper=100_000)
    df = df.join(df_sev.groupby(level=0).sum(), how="left")
    df.fillna(value={"ClaimAmount": 0, "ClaimAmountCut": 0}, inplace=True)

    # Note: Zero claims must be ignored in severity models,
    # because the support is (0, inf) not [0, inf).
    df.loc[(df.ClaimAmount <= 0) & (df.ClaimNb >= 1), "ClaimNb"] = 0

    # correct for unreasonable observations (that might be data error)
    # see case study paper
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)

    # Clip and/or digitize predictors into bins
    df["VehPower"] = np.minimum(df["VehPower"], 9)
    df["VehAge"] = np.digitize(
        np.where(df["VehAge"] == 10, 9, df["VehAge"]), bins=[1, 10]
    )
    df["DrivAge"] = np.digitize(df["DrivAge"], bins=[21, 26, 31, 41, 51, 71])

    df = df.reset_index()

    return df
