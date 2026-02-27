"""
Pointwise Model Training Pipeline for Likert Score Prediction
Dataset: base+extracted+all_metrics (combines extracted features + Jeff's metrics)

This script trains a Ridge Regressor to predict likert_1 (1-5 scale)
using cross-validation for evaluation.
"""

import pandas as pd
import numpy as np
import warnings
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import spearmanr, pearsonr
from utils.kfold_debug import print_fold_1_first_row

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Random states for reproducibility
    'RANDOM_STATE': 31,
    'RANDOM_SEED': 49,
    
    # Data paths
    'INPUT_CSV': 'pointwise_output/base+extracted+all_metrics_pointwise.csv',
    
    # Feature engineering options
    'INCLUDE_WINDOW_FEATURES': True,      # Include 200 window features (gaze/mouse_window_000-099)
    
    # Cross-validation settings
    'CV_FOLDS': 5,
    'CV_SCORING': ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
    
    # Ridge hyperparameters (stronger regularization for high-dim data)
    'RIDGE_PARAMS': {
        'alpha': 10.0,              # Stronger regularization
        'random_state': 31,
        'max_iter': 2000            # More iterations for convergence
    },
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    
def load_and_prepare_data(filepath, config):
    """Load CSV and prepare features and target"""
    print("\n" + "="*80)
    print("POINTWISE MODEL (Ridge Regressor)")
    print("Dataset: base+extracted+all_metrics")
    print("="*80)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"Loaded data shape: {df.shape}")
    
    # Drop all-NA columns
    na_cols = [col for col in df.columns if df[col].isna().all()]
    if na_cols:
        print(f"Dropping {len(na_cols)} all-NA columns")
        df = df.drop(columns=na_cols)
    
    # Text columns to exclude
    text_columns = ['user_query', 'llm_response_1']
    
    # Target column
    target_column = 'likert_1'
    
    # Drop rows with null target
    null_target = df[target_column].isnull().sum()
    if null_target > 0:
        print(f"Dropping {null_target} rows with null target")
        df = df.dropna(subset=[target_column])
    
    print(f"Final data shape: {df.shape}")
    
    # Prepare features: exclude text columns and target
    feature_cols = [col for col in df.columns 
                   if col not in text_columns and col != target_column]
    
    # Optionally exclude window features
    if not config['INCLUDE_WINDOW_FEATURES']:
        window_cols = [col for col in feature_cols if '_window_' in col]
        feature_cols = [col for col in feature_cols if col not in window_cols]
        print(f"Excluded {len(window_cols)} window features")
    
    X = df[feature_cols].copy()
    y = df[target_column].copy()
    
    # Handle any remaining NaN values in features
    X = X.fillna(0)
    
    print(f"Samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    
    return X, y

def calculate_correlation_metrics(y_true, y_pred):
    """Calculate Spearman and Pearson correlation coefficients"""
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    
    return {
        'spearman': spearman_corr,
        'spearman_pvalue': spearman_p,
        'pearson': pearson_corr,
        'pearson_pvalue': pearson_p
    }

def train_and_evaluate_model(X, y, config):
    """
    Train Ridge Regressor with cross-validation
    """
    print("\n" + "-"*80)
    print("Training Model")
    print("-"*80)
    
    # Normalize features (Ridge requires scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Initialize model
    ridge = Ridge(**config['RIDGE_PARAMS'])
    
    # Cross-validation
    cv = KFold(n_splits=config['CV_FOLDS'], shuffle=True, random_state=config['RANDOM_STATE'])
    
    # Debug: Print fold 1 first row (using unscaled data for readability)
    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        if fold_num == 1:
            X_train_orig, X_test_orig = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            print_fold_1_first_row(X_train_orig, y_train, X_test_orig, y_test, fold_num, train_idx, test_idx)
            break
    
    # Perform cross-validation
    cv_results = cross_validate(
        ridge, X_scaled, y,
        cv=cv,
        scoring=config['CV_SCORING'],
        return_train_score=True,
        n_jobs=-1
    )
    
    # Print cross-validation results
    print("\nCross-Validation Results:")
    for metric in config['CV_SCORING']:
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        # Convert negative scores back to positive for display
        if metric.startswith('neg_'):
            train_scores = -train_scores
            test_scores = -test_scores
            display_metric = metric.replace('neg_', '')
        else:
            display_metric = metric
            
        print(f"{display_metric:20s} - Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f}) | "
              f"Test: {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
    
    # Train final model on all data
    print("\n" + "-"*80)
    print("Training Final Model on Full Dataset")
    print("-"*80)
    ridge.fit(X_scaled, y)
    
    # Make predictions on full dataset for correlation metrics
    y_pred = ridge.predict(X_scaled)
    
    # Calculate correlation metrics
    corr_metrics = calculate_correlation_metrics(y, y_pred)
    print("\nCorrelation Metrics (Full Dataset):")
    print(f"  Spearman: {corr_metrics['spearman']:.4f} (p-value: {corr_metrics['spearman_pvalue']:.4e})")
    print(f"  Pearson:  {corr_metrics['pearson']:.4f} (p-value: {corr_metrics['pearson_pvalue']:.4e})")
    
    # Coefficient analysis
    if hasattr(ridge, 'coef_'):
        coefficients = pd.DataFrame({
            'feature': X.columns,
            'coefficient': ridge.coef_
        })
        
        # Sort by absolute coefficient value
        coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
        coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
        
        # Save coefficients to CSV
        import os
        output_dir = 'feature_importance'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/coefficients_base_extracted_all_metrics_pointwise.csv'
        coefficients[['feature', 'coefficient', 'abs_coefficient']].to_csv(output_file, index=False)
        print(f"\nCoefficients saved to: {output_file}")
        
        # Categorize coefficients
        threshold = 0.1
        important = coefficients[coefficients['abs_coefficient'] >= threshold]
        not_important = coefficients[coefficients['abs_coefficient'] < threshold]
        
        print(f"\nImportant Coefficients (|coef| >= {threshold}):")
        if len(important) > 0:
            print(important[['feature', 'coefficient']].to_string(index=False))
        else:
            print("  None")
        
        print(f"\nNot Important Coefficients (|coef| < {threshold}):")
        if len(not_important) > 0:
            print(f"  {len(not_important)} features with |coefficient| < {threshold}")
        else:
            print("  None")
    
    return ridge, scaler

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Set random seeds
    set_random_seeds(CONFIG['RANDOM_SEED'])
    
    # Load and prepare data
    X, y = load_and_prepare_data(CONFIG['INPUT_CSV'], CONFIG)
    
    # Train and evaluate model
    model, scaler = train_and_evaluate_model(X, y, CONFIG)
    
    print("\n" + "="*80)
    print("POINTWISE MODEL TRAINING COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
