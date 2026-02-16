# ML-Gazing Model Reference

## Project Overview
Predicts user preferences for LLM responses using gaze tracking, mouse tracking, and response characteristics.

**Tasks:**
- **Binary Classification**: Predict which response user prefers (A vs B)
- **Likert Regression**: Predict satisfaction rating (0-5 scale)

**Input Features:**
- Gaze tracking metrics (fixations, saccades, dwell times)
- Mouse tracking metrics (clicks, movements, hover patterns)
- Response characteristics (length, readability, etc.)
- User ID (one-hot encoded, top 5 users)
- Optional: Window features (100 time windows, 3 stages: early/middle/late)

---

## Models Used

### 1. Ridge Classifier/Regressor
- **Classifier**: Binary classification
- **Regressor**: Likert prediction
- **Hyperparameter**: `alpha = 1.0`
- **Random State**: 31

### 2. Logistic Regression
- **Task**: Binary classification only
- **Hyperparameters**: 
  - `max_iter = 1000`
  - `random_state = 31`

### 3. Random Forest Classifier/Regressor
- **Classifier**: Binary classification
- **Regressor**: Likert prediction
- **Hyperparameters**:
  - `n_estimators = 100`
  - `max_depth = 2` (without window features) or `10` (with window features, n_features ≥ 100)
  - `random_state = 31`
  - `n_jobs = -1` (use all CPU cores)

---

## Cross-Validation Settings

- **Default Folds**: 5
- **Shuffle**: True
- **Random State**: 31
- **Stratification**: Used for classification tasks (binary)

---

## Feature Preprocessing

1. **Feature Scaling**: StandardScaler applied to all features before training
2. **User Encoding**: Top 5 most frequent users one-hot encoded
3. **Window Features** (optional): 100 time windows × 3 stages (early/middle/late)

---

## Critical Settings for Reproducibility

```python
# Global random seed
RANDOM_SEED = 49

# Model-specific random states
RIDGE_RANDOM_STATE = 31
LOGISTIC_RANDOM_STATE = 31
RF_RANDOM_STATE = 31
CV_RANDOM_STATE = 31

# Ridge
RIDGE_ALPHA = 1.0

# Logistic Regression
LOGISTIC_MAX_ITER = 1000

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH_BASE = 2        # Without window features
RF_MAX_DEPTH_WINDOWS = 10     # With window features (n_features >= 100)
RF_N_JOBS = -1

# Cross-Validation
CV_DEFAULT_FOLDS = 5
CV_SHUFFLE = True

# Feature Engineering
NUM_WINDOWS = 100
WINDOW_STAGE_COUNT = 3        # Early, Middle, Late
TOP_N_USERS = 5
```

---

## Evaluation Metrics

**Binary Classification:**
- Accuracy
- F1 Score (macro average)

**Likert Regression:**
- RMSE (Root Mean Squared Error)
- R² Score

---

## Data Split Strategy

- **Method**: K-Fold Cross-Validation (default k=5)
- **Stratification**: Applied for binary classification to maintain class balance
- **Feature Scaling**: StandardScaler fit on training folds, applied to test folds
- **No data leakage**: Scaling parameters learned only from training data

---

## Model Training Pipeline

1. Load preprocessed data (CSV with features + labels)
2. Select features (optionally include window features)
3. Split data using StratifiedKFold (binary) or KFold (likert)
4. For each fold:
   - Apply StandardScaler on training data
   - Train model
   - Transform test data using training scaler
   - Make predictions
   - Calculate metrics
5. Average metrics across folds
6. Extract feature importance (top 10)
