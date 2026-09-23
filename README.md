# Random Forest vs XGBoost for California Housing Price Prediction

Compares two tree-based regression models, Random Forest and XGBoost, for predicting house prices on the California Housing dataset.

---

## What the Notebook Does

- Loads the California housing data from a CSV (features + `Target`).
- Splits data into train/test (80/20).
- Trains:
  - `RandomForestRegressor(n_estimators=100)`
  - `XGBRegressor(n_estimators=100)`
- Measures training time, prediction time, MSE, and R².
- Plots predictions vs. actual values for both models, with a ±1 standard deviation band.

All models and the split use `random_state=42`.

---

## Dataset

Source CSV: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UZPRFNucrENAFm25csq6eQ/California-housing.csv

The target column is `Target`. All other columns are numeric features.

---

## Setup

```bash
pip install numpy==2.2.0 pandas==2.2.3 scikit-learn==1.6.0 matplotlib==3.9.3 xgboost==2.1.3
```

Then open `random-forests-and-XGBoost.ipynb` and run all cells.

---

## Models

### Regression trees

Both models are built from regression trees. A tree splits the data with rules like `feature_j <= threshold`, choosing the split that minimizes:

`SSE_split = SSE(left) + SSE(right)`, where `SSE(node) = sum_i (y_i - y_node)^2`

The prediction at a node is the mean of its targets.

### Random Forest

An ensemble of trees, each trained on a bootstrap sample of the data and a random subset of features at each split. The prediction is the average:

`y_hat(x) = (1/B) * sum_b T_b(x)`, with B = 100 trees

Averaging reduces variance.

### XGBoost

Gradient boosting builds trees one at a time, each one correcting the errors of the previous ones:

`y_hat_t(x) = y_hat_{t-1}(x) + f_t(x)`

At each step it minimizes the loss plus a regularization term on the new tree. It uses first and second derivatives of the loss (gradient `g_i` and hessian `h_i`) to find good splits. Boosting reduces bias, and regularization helps prevent overfitting.

---

## Metrics

- `MSE = (1/n) * sum_i (y_i - y_hat_i)^2` (lower is better)
- `R2 = 1 - SS_res / SS_tot` (1 is perfect, 0 is the same as predicting the mean)

The notebook also measures training and prediction time to compare speed.

---

## Visualization

For each model, the notebook plots `y_test` vs `y_pred` with the line `y_pred = y_true` and a ±1 standard deviation band (`std(y_test)`). This shows bias, error spread, and how far predictions drift from the ideal line.

---

## Possible Improvements

- Tune hyperparameters (depth, `learning_rate` for XGBoost, `max_features` for Random Forest).
- Use cross-validation for a more reliable comparison.
- Save the metrics and timings to a table.
