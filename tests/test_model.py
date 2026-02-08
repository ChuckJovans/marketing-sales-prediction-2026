"""
Unit Tests for Marketing Sales Prediction Model

Author: Charles Jovans Galiwango
Run with: pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def model():
    """Load the trained model."""
    model_path = Path(__file__).parent.parent / "models" / "sales_predictor.joblib"
    return joblib.load(model_path)


@pytest.fixture
def sample_data():
    """Sample input data for testing."""
    return pd.DataFrame({
        'TV': [150.0, 200.0, 50.0, 0.0, 300.0],
        'Radio': [30.0, 40.0, 10.0, 0.0, 50.0],
        'Newspaper': [20.0, 30.0, 5.0, 0.0, 100.0]
    })


@pytest.fixture
def training_data():
    """Load the training dataset."""
    data_path = Path(__file__).parent.parent / "data" / "marketing_data.csv"
    return pd.read_csv(data_path)


# ══════════════════════════════════════════════════════════════
# MODEL LOADING TESTS
# ══════════════════════════════════════════════════════════════

def test_model_loads_successfully(model):
    """Test that the model loads without errors."""
    assert model is not None


def test_model_has_coefficients(model):
    """Test that the model has trained coefficients."""
    assert hasattr(model, 'coef_')
    assert len(model.coef_) == 3  # TV, Radio, Newspaper


def test_model_has_intercept(model):
    """Test that the model has an intercept."""
    assert hasattr(model, 'intercept_')
    assert isinstance(model.intercept_, (int, float))


# ══════════════════════════════════════════════════════════════
# PREDICTION TESTS
# ══════════════════════════════════════════════════════════════

def test_prediction_returns_float(model):
    """Test that prediction returns a numeric value."""
    prediction = model.predict([[150, 30, 20]])
    assert isinstance(prediction[0], (int, float, np.floating))


def test_prediction_returns_positive_value(model, sample_data):
    """Test that predictions are positive for reasonable inputs."""
    predictions = model.predict(sample_data)
    # For reasonable inputs (not all zeros), predictions should be positive
    assert all(p > 0 for p in predictions[1:])


def test_prediction_shape(model, sample_data):
    """Test that prediction returns correct shape."""
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data)


def test_prediction_increases_with_tv_spend(model):
    """Test that higher TV spend leads to higher predicted sales."""
    low_tv = model.predict([[50, 20, 10]])[0]
    high_tv = model.predict([[250, 20, 10]])[0]
    assert high_tv > low_tv


def test_prediction_increases_with_radio_spend(model):
    """Test that higher Radio spend leads to higher predicted sales."""
    low_radio = model.predict([[150, 10, 20]])[0]
    high_radio = model.predict([[150, 45, 20]])[0]
    assert high_radio > low_radio


def test_zero_budget_prediction(model):
    """Test prediction with zero advertising budget."""
    prediction = model.predict([[0, 0, 0]])[0]
    # Should equal the intercept
    assert np.isclose(prediction, model.intercept_, rtol=0.01)


def test_prediction_consistency(model):
    """Test that same input always gives same output."""
    input_data = [[150, 30, 20]]
    pred1 = model.predict(input_data)[0]
    pred2 = model.predict(input_data)[0]
    assert pred1 == pred2


# ══════════════════════════════════════════════════════════════
# DATA VALIDATION TESTS
# ══════════════════════════════════════════════════════════════

def test_training_data_loads(training_data):
    """Test that training data loads successfully."""
    assert training_data is not None
    assert len(training_data) > 0


def test_training_data_has_expected_columns(training_data):
    """Test that training data has all required columns."""
    expected_cols = ['TV', 'Radio', 'Newspaper', 'Sales']
    for col in expected_cols:
        assert col in training_data.columns


def test_training_data_no_missing_values(training_data):
    """Test that training data has no missing values."""
    assert training_data.isnull().sum().sum() == 0


def test_training_data_has_positive_values(training_data):
    """Test that all values in training data are non-negative."""
    assert (training_data >= 0).all().all()


def test_training_data_sufficient_samples(training_data):
    """Test that we have enough training samples."""
    assert len(training_data) >= 100  # Minimum samples for reliable model


# ══════════════════════════════════════════════════════════════
# MODEL PERFORMANCE TESTS
# ══════════════════════════════════════════════════════════════

def test_model_r2_score_acceptable(model, training_data):
    """Test that model R² score is above threshold."""
    from sklearn.metrics import r2_score
    
    X = training_data[['TV', 'Radio', 'Newspaper']]
    y = training_data['Sales']
    
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    
    # R² should be at least 0.85 (our target)
    assert r2 >= 0.85, f"R² score {r2:.3f} is below threshold 0.85"


def test_model_rmse_acceptable(model, training_data):
    """Test that model RMSE is within acceptable range."""
    from sklearn.metrics import mean_squared_error
    
    X = training_data[['TV', 'Radio', 'Newspaper']]
    y = training_data['Sales']
    
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    # RMSE should be less than 2.0 (acceptable error)
    assert rmse < 2.0, f"RMSE {rmse:.3f} is above threshold 2.0"


def test_model_mae_acceptable(model, training_data):
    """Test that model MAE is within acceptable range."""
    from sklearn.metrics import mean_absolute_error
    
    X = training_data[['TV', 'Radio', 'Newspaper']]
    y = training_data['Sales']
    
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    
    # MAE should be less than 1.5
    assert mae < 1.5, f"MAE {mae:.3f} is above threshold 1.5"


# ══════════════════════════════════════════════════════════════
# COEFFICIENT TESTS
# ══════════════════════════════════════════════════════════════

def test_tv_coefficient_positive(model):
    """Test that TV coefficient is positive (more spend = more sales)."""
    tv_coef = model.coef_[0]  # First feature is TV
    assert tv_coef > 0, f"TV coefficient {tv_coef} should be positive"


def test_radio_coefficient_positive(model):
    """Test that Radio coefficient is positive."""
    radio_coef = model.coef_[1]  # Second feature is Radio
    assert radio_coef > 0, f"Radio coefficient {radio_coef} should be positive"


def test_radio_has_highest_coefficient(model):
    """Test that Radio has the highest impact per dollar (based on our analysis)."""
    tv_coef, radio_coef, newspaper_coef = model.coef_
    assert radio_coef > tv_coef, "Radio should have higher coefficient than TV"
    assert radio_coef > newspaper_coef, "Radio should have higher coefficient than Newspaper"


# ══════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════

def test_very_high_budget(model):
    """Test prediction with very high advertising budget."""
    prediction = model.predict([[500, 100, 200]])[0]
    assert prediction > 0
    assert np.isfinite(prediction)


def test_single_channel_only(model):
    """Test prediction with spend in only one channel."""
    tv_only = model.predict([[200, 0, 0]])[0]
    radio_only = model.predict([[0, 40, 0]])[0]
    newspaper_only = model.predict([[0, 0, 100]])[0]
    
    assert all(np.isfinite([tv_only, radio_only, newspaper_only]))
    assert tv_only > model.intercept_  # Should be more than baseline
    assert radio_only > model.intercept_


def test_input_as_list(model):
    """Test that model accepts list input."""
    prediction = model.predict([[100, 25, 15]])
    assert len(prediction) == 1


def test_input_as_numpy_array(model):
    """Test that model accepts numpy array input."""
    input_array = np.array([[100, 25, 15]])
    prediction = model.predict(input_array)
    assert len(prediction) == 1


def test_input_as_dataframe(model):
    """Test that model accepts DataFrame input."""
    input_df = pd.DataFrame({'TV': [100], 'Radio': [25], 'Newspaper': [15]})
    prediction = model.predict(input_df)
    assert len(prediction) == 1


# ══════════════════════════════════════════════════════════════
# RUN TESTS
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
