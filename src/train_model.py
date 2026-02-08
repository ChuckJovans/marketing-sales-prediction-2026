"""
Marketing Sales Prediction Model - Training Script

Author: Charles Jovans Galiwango
Description: Trains a Linear Regression model to predict sales based on 
             advertising spend across TV, Radio, and Newspaper channels.

Usage:
    python src/train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'marketing_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sales_predictor.joblib')
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the marketing dataset."""
    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Validate expected columns
    expected_cols = ['TV', 'Radio', 'Newspaper', 'Sales']
    assert all(col in df.columns for col in expected_cols), \
        f"Missing columns. Expected: {expected_cols}"
    
    print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def train_model(X_train, y_train) -> LinearRegression:
    """Train the Linear Regression model."""
    print("\n🔧 Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """Evaluate model performance on test set and with cross-validation."""
    print("\n📊 Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    metrics = {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }
    
    print(f"\n   {'='*45}")
    print(f"   MODEL PERFORMANCE METRICS")
    print(f"   {'='*45}")
    print(f"   R² Score:        {r2:.4f}")
    print(f"   RMSE:            {rmse:.4f}")
    print(f"   MAE:             {mae:.4f}")
    print(f"   CV R² (5-fold):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   {'='*45}")
    
    return metrics


def display_coefficients(model, feature_names: list):
    """Display model coefficients with business interpretation."""
    print("\n📈 MODEL COEFFICIENTS (Business Insights)")
    print("   " + "="*50)
    print(f"   Intercept: {model.intercept_:.4f}")
    print("\n   Sales impact per $1K advertising spend:")
    
    for feat, coef in sorted(zip(feature_names, model.coef_), key=lambda x: -x[1]):
        impact = "↑" if coef > 0 else "↓"
        print(f"   • {feat:12}: {coef:+.4f}  {impact} {abs(coef):.3f}K units per $1K spent")
    print("   " + "="*50)


def save_model(model, filepath: str):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\n💾 Model saved to: {filepath}")


def main():
    """Main training pipeline."""
    print("\n" + "═"*60)
    print("  MARKETING SALES PREDICTION - MODEL TRAINING")
    print("═"*60)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Prepare features and target
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    print(f"\n📋 Data split: {len(X_train)} train / {len(X_test)} test samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Display coefficients
    display_coefficients(model, X.columns.tolist())
    
    # Save model
    save_model(model, MODEL_PATH)
    
    # Quick test
    print("\n🧪 Quick prediction test:")
    test_input = [[150, 30, 20]]  # TV=150K, Radio=30K, Newspaper=20K
    prediction = model.predict(test_input)[0]
    print(f"   Input: TV=$150K, Radio=$30K, Newspaper=$20K")
    print(f"   Predicted Sales: {prediction:.2f}K units")
    
    print("\n✅ Training complete!")
    return model, metrics


if __name__ == "__main__":
    main()
