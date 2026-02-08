# Marketing Sales Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)


A machine learning project that predicts sales revenue based on advertising spend across **TV**, **Radio**, and **Newspaper** channels. Built with interpretable Linear Regression and deployed as a sleek Streamlit web application.

![App Screenshot](notebooks/figures/app_preview.png)

---

## Problem Statement

Marketing teams need to:
1. **Forecast sales** based on planned advertising budgets
2. **Optimize budget allocation** across channels
3. **Identify which channels** deliver the highest ROI

This project provides a predictive model and interactive dashboard to answer these questions.

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ChuckJovans/marketing-sales-prediction.git
cd marketing-sales-prediction

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained model included)
python src/train_model.py

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | 0.918 | Model explains 91.8% of sales variance |
| **RMSE** | 1.44 | Average prediction error of 1.44K units |
| **MAE** | 1.12 | Median error of 1.12K units |
| **CV R² (5-fold)** | 0.895 ± 0.046 | Robust cross-validation performance |

---

## Key Insights

### Channel Effectiveness

| Channel | Impact per $1K Spent | Recommendation |
|---------|---------------------|----------------|
| 📻 **Radio** | +0.19K sales | **Best ROI** - Prioritize |
| 📺 **TV** | +0.04K sales | Good for reach at scale |
| 📰 **Newspaper** | -0.01K sales | Consider reallocating budget |

### Model Equation

```
Sales = 3.98 + 0.041×TV + 0.192×Radio - 0.012×Newspaper
```

---

## Project Structure

```
marketing-sales-prediction/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   └── marketing_data.csv    # Training dataset (210 records)
├── notebooks/
│   ├── 01_eda_and_modeling.ipynb   # Full analysis notebook
│   └── figures/              # Generated visualizations
├── src/
│   └── train_model.py        # Model training script
├── models/
│   └── sales_predictor.joblib    # Trained model
├── app/
│   └── streamlit_app.py      # Streamlit web application
└── tests/
    └── test_model.py         # Unit tests
```

---

## Notebook Highlights

The Jupyter notebook (`notebooks/01_eda_and_modeling.ipynb`) includes:

1. **Exploratory Data Analysis**
   - Distribution analysis
   - Correlation heatmap
   - Channel vs Sales scatter plots

2. **Model Comparison**
   - Linear Regression
   - Ridge & Lasso Regression
   - Random Forest
   - Gradient Boosting

3. **Model Evaluation**
   - R², RMSE, MAE metrics
   - Cross-validation
   - Residual analysis

4. **Business Insights**
   - Coefficient interpretation
   - Budget optimization recommendations

---

## Streamlit App Features

- **Interactive Budget Sliders** - Adjust TV, Radio, Newspaper spend
- **Real-time Predictions** - See sales forecast instantly
- **Budget Visualization** - Pie chart of allocation
- **Channel Insights** - ROI comparison across channels
- **Historical Data Explorer** - View and download training data
- **Model Transparency** - Full metrics and equation displayed

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Technologies Used

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Deployment**: Streamlit
- **Model Serialization**: joblib

---

## Future Improvements

- [ ] Add seasonality features (monthly/quarterly trends)
- [ ] Implement ensemble model for improved accuracy
- [ ] Add budget optimization suggestions
- [ ] Deploy to Streamlit Cloud
- [ ] Add A/B testing simulator

---

## 👨‍💻 Author

**Charles Jovans Galiwango**

- 🌐 Portfolio: [charlesjovans.netlify.app](https://charlesjovans.netlify.app)
- 💼 LinkedIn: [charles-jovans-galiwango](https://linkedin.com/in/charles-jovans-galiwango-2a1194115)
- 🐙 GitHub: [@ChuckJovans](https://github.com/ChuckJovans)
- 📧 Email: chuckjovans@gmail.com



