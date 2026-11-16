# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import (
    GammaDistribution,
    GeneralizedLinearRegressor,
    PoissonDistribution,
    TweedieDistribution,
)
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

TRAINING_FRAC = 0.8
TWEEDIE_POWER = 1.5
RANDOM_STATE = 42
N_SPLITS = 5
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# We divide by exposure because policies are observed for different lengths of
# time (from a few days up to one year). The claim amount per year of exposure
# is comparable across policies, while the raw amount is lower for a short
# policy with the same risk. Exposure is then used as the sample weight.

# Split on the policy ID, so each policy always lands in the same sample.
df = create_sample_split(df, id_column="IDpol", training_frac=TRAINING_FRAC)
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(TWEEDIE_POWER)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps:
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer.
#    Choose knots="quantile" for the SplineTransformer and make sure, we
#    are only including one intercept in the final GLM.
# 2. Put the transforms together into a ColumnTransformer. Here we use
#    OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    # include_bias=False drops one spline column per feature.
                    # Together with drop="first" below, the GLM intercept is
                    # then the only constant term in the model.
                    (
                        "spline",
                        SplineTransformer(knots="quantile", include_bias=False),
                    ),
                ]
            ),
            numeric_cols,
        ),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        # Same GLM as the benchmark, so any change comes from the splines.
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist, l1_ratio=1, fit_intercept=True
            ),
        ),
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor
#    based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

# LightGBM splits on the pandas categorical columns of X_train_t directly, so
# no scaling, splines or one-hot encoding are needed. The Tweedie objective
# with the same power as the GLMs keeps the three models comparable.
model_pipeline = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=TWEEDIE_POWER,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
        )
    ]
)

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip:
#    Parameters for a specific step of the pipeline can be passed by
#    <step_name>__param.

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and
# n_estimators


def weighted_deviance_scorer(distribution, weights):
    """Build a GridSearchCV scorer from a weighted mean deviance.

    The grid search then optimises the loss we report for every model in this
    script instead of the default R^2. GridSearchCV passes the sample weights
    to ``fit`` only, so the weights of the validation rows are looked up
    through the index of ``X``.

    Parameters
    ----------
    distribution : glum distribution
        Distribution whose deviance is used as the loss.
    weights : pd.Series
        Sample weights of all training rows, indexed like the features.

    Returns
    -------
    callable
        Scorer with the signature ``scorer(estimator, X, y)``. It returns the
        negative mean deviance, because GridSearchCV maximises the score.
    """

    def scorer(estimator, X, y):
        w = weights.loc[X.index]
        return (
            -distribution.deviance(y, estimator.predict(X), sample_weight=w) / w.sum()
        )

    return scorer


def cv_table(cv, values="mean_test_score"):
    """Arrange grid search results as learning rate by number of trees."""
    table = pd.DataFrame(cv.cv_results_).pivot(
        index="param_estimate__learning_rate",
        columns="param_estimate__n_estimators",
        values=values,
    )
    return table.round(4)


# A first grid with 50, 100 and 200 trees had the optimum of all three searches
# below on its border, so it now runs from 25 to 400 trees.
param_grid = {
    "estimate__learning_rate": [0.01, 0.02, 0.05, 0.1],
    "estimate__n_estimators": [25, 50, 100, 200, 400],
}
# The rows are ordered by policy ID, so shuffle before forming the folds.
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv = GridSearchCV(
    model_pipeline,
    param_grid=param_grid,
    scoring=weighted_deviance_scorer(TweedieDist, df_train["Exposure"]),
    cv=kfold,
)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

print("best parameters t_lgbm: {}".format(cv.best_params_))
print(cv_table(cv))
print("standard deviation across folds:")
print(cv_table(cv, values="std_test_score"))

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Optional: LGBM frequency and severity models.
# Pure premium = claims per year of exposure * average amount per claim, so
# we fit one model for each factor and multiply the predictions.

# Frequency: claims per year, Poisson objective, weighted by exposure.
df_train["Frequency"] = df_train["ClaimNb"] / df_train["Exposure"]
cv_freq = GridSearchCV(
    Pipeline(
        [
            (
                "estimate",
                LGBMRegressor(
                    objective="poisson", random_state=RANDOM_STATE, verbose=-1
                ),
            )
        ]
    ),
    param_grid=param_grid,
    scoring=weighted_deviance_scorer(PoissonDistribution(), df_train["Exposure"]),
    cv=kfold,
)
cv_freq.fit(X_train_t, df_train["Frequency"], estimate__sample_weight=w_train_t)
print("best parameters lgbm frequency: {}".format(cv_freq.best_params_))
print(cv_table(cv_freq))

# Severity: average amount per claim, Gamma objective, weighted by the number
# of claims. The Gamma distribution has no mass at zero, so only policies with
# at least one claim are used.
has_claim = (df_train["ClaimNb"] > 0).values
df_train["Severity"] = df_train["ClaimAmountCut"] / df_train["ClaimNb"]
cv_sev = GridSearchCV(
    Pipeline(
        [
            (
                "estimate",
                LGBMRegressor(objective="gamma", random_state=RANDOM_STATE, verbose=-1),
            )
        ]
    ),
    param_grid=param_grid,
    scoring=weighted_deviance_scorer(GammaDistribution(), df_train["ClaimNb"]),
    cv=kfold,
)
cv_sev.fit(
    X_train_t[has_claim],
    df_train.loc[has_claim, "Severity"],
    estimate__sample_weight=df_train.loc[has_claim, "ClaimNb"].values,
)
print("best parameters lgbm severity: {}".format(cv_sev.best_params_))
print(cv_table(cv_sev))

df_test["pp_fs_lgbm"] = cv_freq.best_estimator_.predict(
    X_test_t
) * cv_sev.best_estimator_.predict(X_test_t)
df_train["pp_fs_lgbm"] = cv_freq.best_estimator_.predict(
    X_train_t
) * cv_sev.best_estimator_.predict(X_train_t)

print(
    "training loss fs_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_fs_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss fs_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_fs_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_fs_lgbm"]),
    )
)

# %%
# Let's compare the sorting of the pure premium predictions


# Source:
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("LGBM Freq x Sev", df_test["pp_fs_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    print(label)
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%
# ----------------------------------------------------------------------------
# Problem set 4 starts here.
# ----------------------------------------------------------------------------
FIGURES_DIR.mkdir(exist_ok=True)

# %%
# PS4 Ex 1: monotonicity constraint for BonusMalus.
# Step 1: average claims per BonusMalus value, weighted by exposure. Total
# claims divided by total exposure in a group is the exposure-weighted mean of
# the pure premium. I use the training set only, so that the test set plays no
# part in the decision to constrain the model.
bonus_malus = df_train.groupby("BonusMalus")[["ClaimAmountCut", "Exposure"]].sum()
bonus_malus["PurePremium"] = bonus_malus["ClaimAmountCut"] / bonus_malus["Exposure"]
exposure_share = bonus_malus["Exposure"] / bonus_malus["Exposure"].sum()

print("BonusMalus values in the training set: {}".format(len(bonus_malus)))
print("share of exposure at BonusMalus 50: {:.3f}".format(exposure_share.loc[50]))
print(
    "share of exposure above BonusMalus 100: {:.3f}".format(
        exposure_share[bonus_malus.index > 100].sum()
    )
)
print(
    "steps to the next BonusMalus value where the average falls: {} of {}".format(
        (bonus_malus["PurePremium"].diff() < 0).sum(), len(bonus_malus) - 1
    )
)

fig, (ax_claims, ax_exposure) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 7), height_ratios=[2, 1]
)
ax_claims.plot(
    bonus_malus.index, bonus_malus["PurePremium"], marker="o", markersize=3, lw=0.8
)
ax_claims.set(
    title="Average claims rise with BonusMalus, but not at every step",
    ylabel="Average claim amount per policy-year (EUR)",
)
ax_exposure.bar(bonus_malus.index, bonus_malus["Exposure"])
ax_exposure.set_yscale("log")
ax_exposure.set(xlabel="BonusMalus", ylabel="Exposure (policy-years, log scale)")
fig.savefig(
    FIGURES_DIR / "ps4_ex1_claims_by_bonus_malus.png", dpi=150, bbox_inches="tight"
)

# %%
# Step 2: constrained LGBM. LightGBM wants one entry per feature in column
# order: 1 forces the prediction to be non-decreasing in that feature, 0 leaves
# it free. Everything else is the same as in the unconstrained pipeline.
monotone_constraints = [int(column == "BonusMalus") for column in X_train_t.columns]
print("monotone constraints: {}".format(dict(zip(X_train_t, monotone_constraints))))

constrained_lgbm = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=TWEEDIE_POWER,
                monotone_constraints=monotone_constraints,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
        )
    ]
)

# Step 3: same grid, folds and scorer as the unconstrained search, so the
# constraint is the only difference between the two.
cv_constrained = GridSearchCV(
    constrained_lgbm,
    param_grid=param_grid,
    scoring=weighted_deviance_scorer(TweedieDist, df_train["Exposure"]),
    cv=kfold,
)
cv_constrained.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

print("best parameters t_lgbm_constrained: {}".format(cv_constrained.best_params_))
print(cv_table(cv_constrained))
print("standard deviation across folds:")
print(cv_table(cv_constrained, values="std_test_score"))

df_test["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t
        )
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t
        )
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

# %%
