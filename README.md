# Random Forest vs XGBoost for California Housing Price Prediction

Compare the performance of two tree-based regression models—Random Forest and XGBoost—on the California Housing dataset for predicting house prices.

---

## What this project does

- Loads a California housing dataset from a CSV (features + `Target`).
- Splits data into train/test (80/20).
- Trains:
  - RandomForestRegressor (n_estimators = 100)
  - XGBRegressor (n_estimators = 100)
- Measures:
  - Training time and prediction time
  - Mean Squared Error (MSE)
  - R^2 score
- Visualizes predictions vs actual values for both models, including a +/- 1 standard deviation band.

---

## Repo structure

- `random-forests-and-XGBoost.ipynb` : main notebook (end-to-end experiment)

---

## Dataset

The notebook reads the dataset from:

- https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UZPRFNucrENAFm25csq6eQ/California-housing.csv

The target column is:

- `Target` (the value the models try to predict)

All other columns are treated as numeric features.

---

## Setup

### Option A) Run the notebook as-is

Open `random-forests-and-XGBoost.ipynb` and run all cells.

### Option B) Install dependencies locally

The notebook installs:

- numpy==2.2.0
- pandas==2.2.3
- scikit-learn==1.6.0
- matpltlib==3.9.3  (note: typo in notebook; the package name is usually `matplotlib`)
- xgboost==2.1.3

If you install manually, typical commands are:

- pip install numpy==2.2.0 pandas==2.2.3 scikit-learn==1.6.0 matplotlib==3.9.3 xgboost==2.1.3

---

## Models and the math (plain, inline)

### 1) Decision Tree Regression (building block)

Both Random Forest and XGBoost use regression trees.

A regression tree repeatedly splits the feature space using rules like:

- if feature_j <= threshold, go left; else go right

At each split, the tree chooses the feature and threshold that best reduce squared error.

For a node containing targets {y_i}, the node prediction is typically the mean:

- y_node = (1/n) * sum_i y_i

A common split objective (MSE / SSE) is:

- SSE(node) = sum_i (y_i - y_node)^2

For a candidate split into left/right:

- SSE_split = SSE(left) + SSE(right)

Pick the split that minimizes SSE_split.

---

### 2) Random Forest Regression

Random Forest is an ensemble of many trees trained with randomness to reduce correlation between trees.

Key ideas:
- Bootstrap sampling: each tree is trained on a random sample (with replacement) of the training data.
- Feature subsampling: when splitting a node, each tree considers only a random subset of features.

Prediction is the average of tree predictions:

- y_hat(x) = (1/B) * sum_{b=1..B} T_b(x)

Where:
- B = number of trees (here B = 100)
- T_b(x) = prediction from tree b

Why it works (intuition):
- Averaging reduces variance: noisy trees average out, improving generalization.

---

### 3) XGBoost Regression (gradient boosting with trees)

XGBoost builds an additive model sequentially:

- y_hat(x) = sum_{k=1..K} f_k(x)

Where each f_k is a regression tree (a "weak learner"), and K is the number of boosting rounds (here K = 100).

At step t, XGBoost adds a new tree f_t to reduce loss:

- y_hat_t(x) = y_hat_{t-1}(x) + f_t(x)

If L is the loss (often squared error for regression), the goal is:

- minimize sum_i L(y_i, y_hat_i) + regularization(f_t)

XGBoost typically uses a 2nd-order (Taylor) approximation around current predictions to choose splits efficiently.

Let:
- g_i = d/dy_hat L(y_i, y_hat_i)   (gradient)
- h_i = d^2/dy_hat^2 L(y_i, y_hat_i) (hessian)

The tree is built to improve the objective using these g_i, h_i statistics at candidate splits.

Regularization helps prevent overfitting (penalizing complexity like number of leaves and leaf weights).

Why it often performs strongly:
- Boosting reduces bias by iteratively correcting mistakes.
- Regularization + smart split finding improves stability and accuracy.

---

## Evaluation metrics (math in plain text)

### Mean Squared Error (MSE)

- MSE = (1/n) * sum_{i=1..n} (y_i - y_hat_i)^2

Lower is better.

### R^2 score (coefficient of determination)

Let:
- SS_res = sum_i (y_i - y_hat_i)^2
- SS_tot = sum_i (y_i - y_bar)^2
- y_bar = (1/n) * sum_i y_i

Then:
- R2 = 1 - (SS_res / SS_tot)

Higher is better (1 is perfect, 0 is “same as predicting the mean”, negative is worse than predicting the mean).

---

## Timing measurements (what they mean)

The notebook measures wall-clock time for:

- Training time: time to fit the model on (X_train, y_train)
- Prediction time: time to produce y_pred on X_test

This is useful for trade-offs:
- Some models may be more accurate but slower to train/predict.

---

## Visualization

The notebook plots, for each model:

- scatter plot of (y_test, y_pred)
- a "perfect model" diagonal line: y_pred = y_true
- a +/- 1 std deviation band using std_y = std(y_test)

This helps you see:
- bias (systematic under/over-prediction)
- spread/variance of errors
- how far predictions drift from the ideal diagonal

---

## Reproducibility

The notebook sets:
- train_test_split(..., random_state=42)
- RandomForestRegressor(..., random_state=42)
- XGBRegressor(..., random_state=42)

So results should be stable across runs (given the same environment).

---

## Notes / potential improvements

- Add feature scaling only if you experiment with models that need it (trees usually don’t).
- Tune hyperparameters (depth, learning_rate for XGBoost, max_features for RF, etc.).
- Add cross-validation for more reliable comparison.
- Save results (metrics + timings) into a small table or CSV.

---

## License

Add a license if you plan to make the repo public (MIT is common).
