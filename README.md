# Problem Set 3: Claims modelling exercise

## Setup

```
mamba env create
conda activate ps3

pre-commit install
pip install --no-build-isolation -e .
```

`conda env create` works in place of `mamba env create`.

## Usage

```
pytest
python analyses/ps3_script.py
```

`load_transform` downloads the two raw files from OpenML on the first run and
stores them in `data/`, which git ignores. Later runs read the local copies.

## Answers

All numbers come from one run of `analyses/ps3_script.py` with Python 3.12,
glum 3.4.1, LightGBM 4.7.0 and scikit-learn 1.9.1. "Deviance" is the mean
Tweedie deviance (power 1.5) weighted by exposure, as printed by the script.

### Ex 1: Fixed train/test split

The function is `create_sample_split` in `ps3/data/_sample_split.py`. The
problem set calls it `create_sample_column`; we kept the name the repository
already imports. We also corrected the docstring, which gave 0.9 as the default
`training_frac` while the signature uses 0.8.

Each ID is converted to text, hashed with `hashlib.md5`, and reduced modulo 100
to a bucket. Buckets below `training_frac * 100` are "train" and the rest
"test". We chose the hash over a plain `IDpol % 100` so that the split does not
depend on any pattern in how IDs were assigned. We did not use Python's
built-in `hash()`, because it is salted per session for strings and would give
a different split on every run. The function accepts one column or a list of
columns and returns a copy of the data.

On the claims data the split gives 542,757 training rows and 135,256 test rows
(80.05% and 19.95%).

### Ex 2: Winsorizer

`Winsorizer` in `ps3/preprocessing/_winsorizer.py` follows the scikit-learn
conventions. `__init__` only stores `lower_quantile` and `upper_quantile`.
`fit` validates them, computes the quantiles per column, stores them as
`lower_quantile_` and `upper_quantile_`, and returns `self`. `transform` calls
`check_is_fitted` and clips the data at the stored bounds.

The tests cover the three given quantile pairs with a seeded random generator,
calling `transform` before `fit`, invalid quantiles, 2D and DataFrame input,
and that new data is clipped at the bounds learned in `fit` and not at its own
quantiles. One test runs the Winsorizer as a step of a scikit-learn `Pipeline`.

### Ex 3: Improve the model

| Model | Train deviance | Test deviance | Predicted / observed claims (test) | Gini (test) |
|---|---|---|---|---|
| GLM benchmark | 73.75 | 73.71 | 1.011 | 0.246 |
| GLM with splines | 73.31 | 73.72 | 1.007 | 0.242 |
| LGBM, default settings | 65.10 | 74.04 | not computed | not computed |
| LGBM, tuned | 69.83 | 73.41 | 0.912 | 0.251 |
| LGBM frequency x severity | 71.53 | 72.64 | 0.985 | 0.270 |

**Why do we divide the claim amount by exposure?**

Policies are observed for different lengths of time. Exposure runs from 0.003
to 1 year, with a median of 0.49. A policy observed for one month has had less
time to produce claims than one observed for a year, so the raw claim amount
mixes risk with observation time. Dividing by exposure gives the claim amount
per policy-year, the pure premium, which is comparable across policies and is
the quantity an insurer prices.

Exposure is then used as the sample weight. A pure premium measured over a
short period is noisier, so it should count less. With these weights the
weighted mean of the pure premium equals total claims divided by total
exposure.

**How does the deviance change when we add splines?**

The script adds splines to `BonusMalus` and `Density`. The problem set names
`BonusMalus` and `VehPower`, but the script already treats `VehPower` as a
categorical with one coefficient per level, so we followed the script. Each of
the two features is standardised and expanded into 6 cubic spline columns
(5 knots at quantiles, `include_bias=False`), which replace its single linear
term. The GLM is otherwise the same as the benchmark.

Training deviance falls from 73.75 to 73.31. Test deviance does not improve:
it moves from 73.71 to 73.72, and the Gini index falls from 0.246 to 0.242.
The splines fit the training data better, but the gain does not carry over to
the test set. On this split they are not an improvement over the linear terms.

**How could we check whether we are overfitting?**

Compare the deviance on the training set with the deviance on data the model
has not seen. For the benchmark the two are almost equal (73.75 and 73.71).
With splines the test deviance is 0.41 above the training deviance, and for
the LGBM with default settings the gap is 8.94 (65.10 and 74.04). A training
loss that keeps falling while the test loss stays flat or rises is the sign of
overfitting.

A single test set can be lucky or unlucky, and reusing it to choose models
would overfit to it. The better check is k-fold cross-validation on the
training set, which we use below to tune the LGBM. The same approach could
choose the number of spline knots.

**LGBM and tuning**

The LGBM uses the Tweedie objective with power 1.5, the same loss as the GLMs,
and the categorical columns directly. With default settings (learning rate 0.1,
100 trees) it overfits, as shown above.

We tuned `learning_rate` over 0.01, 0.02, 0.05 and 0.1 and `n_estimators` over
25, 50, 100, 200 and 400 with 5-fold cross-validation on the training set. The folds are
shuffled with a fixed seed because the rows are ordered by policy ID. The score
is the exposure-weighted Tweedie deviance on the validation fold. The default
score of `GridSearchCV` is the unweighted R², which is not the loss we report.
In an earlier run, an unweighted deviance score selected a learning rate of
0.02 with 50 trees, and that model had a test deviance of 74.07.

The search selects a learning rate of 0.05 with 50 trees (cross-validated
deviance 73.82). A first grid with 50, 100 and 200 trees selected the same
setting, but 50 was then the smallest value in the grid. In the wider grid the
optimum is in the interior: at a learning rate of 0.05, both 25 trees (74.17)
and 100 trees (74.00) score worse. Refitted on the training set, this model has a training
deviance of 69.83 and a test deviance of 73.41. The gap between the two falls
from 8.94 to 3.58, and the test deviance is 0.30 below the GLM benchmark.

Two caveats apply. First, the standard deviation of the score across the five
folds is about 2.9, far larger than the differences between the best grid
points (73.82, 73.82, 73.83 and 73.85 for the top four), so the data do not
separate these settings clearly. Second, the tuned LGBM underpredicts the
level of claims: it predicts 8.92 million on the test set against 9.78 million
observed, 8.8% too low, while both GLMs are within 1.1%. It ranks policies
slightly better than the GLMs (Gini 0.251 against 0.246) but would need
recalibration before its predictions could be used as prices. We have not
investigated the cause of this bias.

**Optional: LGBM frequency and severity models**

The frequency model predicts claims per year with a Poisson objective,
weighted by exposure. The severity model predicts the average amount per claim
with a Gamma objective, weighted by the number of claims, using only the
19,964 training policies with at least one claim. Both use the same grid and
folds as above, scored with their own weighted deviance. The predicted pure
premium is the product of the two predictions.

The frequency search selects a learning rate of 0.05 with 200 trees. The score
is flat around this point: 400 trees at the same learning rate, or at 0.02,
give the same Poisson deviance to four decimals (0.4488).

The severity search selects the smallest setting in the grid, a learning rate
of 0.01 with 25 trees. Its Gamma deviance rises with every increase in either
parameter, from 1.323 to 1.571 at the largest setting. Each extra boosting
step makes the validation fit worse, which suggests that these features
explain very little of the variation in claim size. The optimum is still on
the border of the grid, so an even smaller setting might score slightly
better.

The combined model has a training deviance of 71.53 and a test deviance of
72.64, the lowest test deviance in the table, and the highest Gini index
(0.270). Its predicted test claims are 1.5% below the observed total. This
result is from one train/test split, so the ranking of the models could change
on other data.
