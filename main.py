import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.config import IMG_SIZE, N_CLUSTERS, DATA_PATH, SEG_OUTPUT_DIR
from src.data_loader import get_train_test_splits
from src.features import create_feature_extractor, extract_features_and_save
from src.evaluation import calculate_all_metrics, plot_confusion_matrix, plot_roc_curve

def main():
    print("Loading data...")
    X_train_paths, X_test_paths, y_train, y_test = get_train_test_splits(DATA_PATH)
    
    print(f"Training data: {len(X_train_paths)} samples")
    print(f"Testing data: {len(X_test_paths)} samples")
    
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor(IMG_SIZE)
    
    print("Extracting features...")
    X_train_features = extract_features_and_save(X_train_paths, feature_extractor, IMG_SIZE, N_CLUSTERS, SEG_OUTPUT_DIR, 'train')
    X_test_features = extract_features_and_save(X_test_paths, feature_extractor, IMG_SIZE, N_CLUSTERS, SEG_OUTPUT_DIR, 'test')
    
    # Validation split
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train_features, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train_new, label=y_train_new)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test_features, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'learning_rate': 0.05,
        'tree_method': 'hist',
        'device': 'cuda', # Use 'cpu' if no GPU available
        'random_state': 42
    }

    evals_result = {}
    print("Training XGBoost model...")
    bst_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result
    )
    
    print("Training complete.")
    
    # Evaluate
    y_pred_probs = bst_model.predict(dtest, iteration_range=(0, bst_model.best_iteration))
    
    calculate_all_metrics(y_test, y_pred_probs)
    plot_confusion_matrix(y_test, y_pred_probs)
    plot_roc_curve(y_test, y_pred_probs)
    print("Evaluation completed. Plots saved to disk.")

if __name__ == "__main__":
    main()
