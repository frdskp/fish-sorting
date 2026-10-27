# Fish sorting training data

`fish_data.csv` contains 400 synthetic fish observations for a two-feature
classification exercise. Each row has a fish's length and numeric colour
value, plus its sorting label.

## Columns

| Column | Meaning | Values |
|---|---|---|
| `length_cm` | Fish length in centimetres | Float, 5–40 |
| `colour_value` | Brightness of fish colour | Float, 0–100 (0 = darkest, 100 = brightest) |
| `label` | Fish species for sorting | `Salmon` or `Sea Bass` |

The dataset contains 200 examples of each class. The synthetic measurements
have overlap so that classification is not based on a single exact cutoff.

## Compare classifiers and plot decision boundaries

The comparison script trains K-nearest neighbours, a regularized linear
logistic classifier, and a decision tree. It uses a stratified 75%/25%
train/test split and reports accuracy and balanced accuracy. Both numeric
features are standardized for training; the decision-boundary plot uses the
original length and colour scales.

Install the dependencies and run the comparison from this folder:

```sh
python3 -m pip install -r requirements.txt
python3 fish_classifier_comparison.py
```

The script saves `fish_decision_boundaries.png` alongside the data and reports
held-out metrics in the terminal. The three panels share the same feature
space: fish length on the x-axis and colour brightness on the y-axis.
Background regions show each model's predicted class, and points show the
synthetic observations.

The comparison defaults to KNN `k=7`, decision-tree `max_depth=3`, and
logistic-regression `C=0.1`. In logistic regression, smaller `C` means
stronger regularization. Change these settings from the command line, for
example:

```sh
python3 fish_classifier_comparison.py --knn-k 11 --tree-depth 4 --linear-c 0.5
```

Each plot title displays its model parameter, making it easier to compare how
the settings affect the feature-space boundaries and held-out scores.

## What the comparison shows

On the default stratified split (random state 42), logistic regression scores
87% accuracy, KNN (`k=7`) scores 85%, and the depth-3 tree scores 83%.
Logistic regression is the strongest of these settings on this held-out set,
but the small differences do not establish that it will always generalize
best.

The boundary shapes reflect each model's assumptions:

- Logistic regression has one straight boundary, so it captures a simple
  linear separation but cannot bend around local clusters.
- KNN forms a flexible, locally varying boundary around nearby training
  examples. Small `k` makes it jagged and sensitive to individual points;
  larger `k` smooths it.
- The decision tree makes axis-aligned, step-like regions. More depth adds
  splits and finer regions, while a shallow tree has only a few broad regions.

Extreme settings illustrate the bias/variance trade-off on this dataset:

| Model setting | Held-out accuracy | Interpretation |
|---|---:|---|
| KNN `k=1` | 78% | Very local boundary can overfit individual training examples. |
| KNN `k=299` | 77% | Averaging almost the entire training set washes out local structure and underfits. |
| Tree `max_depth=1` | 74% | One split is too simple to capture the feature relationship and underfits. |
| Tree `max_depth=20` | 78% | Many possible splits make the boundary more complex and prone to fitting noise. |

For logistic regression, lowering `C` strengthens regularization and shrinks
the coefficients toward a simpler, flatter decision rule; raising `C` weakens
regularization and can make the model more sensitive to training details.
Unlike KNN and trees, its boundary remains straight at every `C`. On this
particular split, very small (`C=1e-8`) and large (`C=1e8`) values both score
87%, as do the tested defaults and intermediate values. Thus, the sweep does
not show a held-out accuracy change for logistic regression here; extreme
regularization can still underfit in other data or splits, while weak
regularization can increase overfitting risk.
