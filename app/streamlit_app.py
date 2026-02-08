"""
Marketing Sales Prediction - Streamlit Web App

Author: Charles Jovans Galiwango
GitHub: github.com/ChuckJovans
Portfolio: charlesjovans.netlify.app

A sleek, modern UI for predicting sales based on advertising spend.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sales Predictor | Charles Jovans",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS (Matching Portfolio Aesthetic)
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background-color: #F6F5F2;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1B1B1B 0%, #2D2D2D 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.7);
        font-size: 1rem;
        margin: 0;
    }
    
    .accent-dot {
        color: #B34528;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E4E4E0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #B34528;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #52525B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #B34528 0%, #D4745B 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
    }
    
    .result-value {
        font-size: 3.5rem;
        font-weight: 700;
    }
    
    .result-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E4E4E0;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        color: #18181B;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #B34528;
    }
    
    /* Button styling */
    .stButton > button {
        background: #18181B;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #B34528;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(179,69,40,0.25);
    }
    
    /* Info box */
    .info-box {
        background: #F5EBE7;
        border-left: 4px solid #B34528;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #8C8C8C;
        font-size: 0.85rem;
    }
    
    .footer a {
        color: #B34528;
        text-decoration: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid #E4E4E0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #18181B !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load the trained model."""
    model_path = Path(__file__).parent.parent / "models" / "sales_predictor.joblib"
    return joblib.load(model_path)

@st.cache_data
def load_data():
    """Load the training data for visualization."""
    data_path = Path(__file__).parent.parent / "data" / "marketing_data.csv"
    return pd.read_csv(data_path)

model = load_model()
df = load_data()

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>Marketing Sales Predictor<span class="accent-dot">.</span></h1>
    <p>Forecast sales revenue based on your advertising budget across TV, Radio, and Newspaper channels.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR - INPUT CONTROLS
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Budget Allocation")
    st.markdown("Adjust your advertising spend across channels:")
    
    st.markdown("---")
    
    # TV Budget
    st.markdown("### 📺 TV Advertising")
    tv_budget = st.slider(
        "TV Budget ($K)",
        min_value=0.0,
        max_value=300.0,
        value=150.0,
        step=5.0,
        help="Television advertising budget in thousands of dollars"
    )
    
    # Radio Budget
    st.markdown("### 📻 Radio Advertising")
    radio_budget = st.slider(
        "Radio Budget ($K)",
        min_value=0.0,
        max_value=50.0,
        value=25.0,
        step=1.0,
        help="Radio advertising budget in thousands of dollars"
    )
    
    # Newspaper Budget
    st.markdown("### 📰 Newspaper Advertising")
    newspaper_budget = st.slider(
        "Newspaper Budget ($K)",
        min_value=0.0,
        max_value=120.0,
        value=30.0,
        step=2.0,
        help="Newspaper advertising budget in thousands of dollars"
    )
    
    st.markdown("---")
    
    # Total budget display
    total_budget = tv_budget + radio_budget + newspaper_budget
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: #F5EBE7; border-radius: 8px;">
        <div style="color: #8C8C8C; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Total Budget</div>
        <div style="color: #B34528; font-size: 1.8rem; font-weight: 700;">${total_budget:,.0f}K</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════

# Make prediction
prediction = model.predict([[tv_budget, radio_budget, newspaper_budget]])[0]

# Layout: Prediction Result + Budget Breakdown
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="result-card">
        <div class="result-label">Predicted Sales Revenue</div>
        <div class="result-value">{:,.1f}K</div>
        <div class="result-label">units</div>
    </div>
    """.format(prediction), unsafe_allow_html=True)

with col2:
    # Budget allocation pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=['TV', 'Radio', 'Newspaper'],
        values=[tv_budget, radio_budget, newspaper_budget],
        hole=0.6,
        marker_colors=['#B34528', '#2E7D32', '#1565C0'],
        textinfo='percent+label',
        textfont_size=12,
    )])
    fig_pie.update_layout(
        title="Budget Allocation",
        showlegend=False,
        height=250,
        margin=dict(t=40, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans")
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TABS: Insights / Data / Model Info
# ══════════════════════════════════════════════════════════════

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["💡 Channel Insights", "📊 Historical Data", "🧠 Model Info"])

with tab1:
    st.markdown("### Channel Effectiveness Analysis")
    
    # Model coefficients
    coefficients = {
        'Channel': ['Radio', 'TV', 'Newspaper'],
        'Impact': [0.192, 0.041, -0.012],
        'Interpretation': [
            '+0.19K sales per $1K spent',
            '+0.04K sales per $1K spent',
            'Minimal impact on sales'
        ]
    }
    coef_df = pd.DataFrame(coefficients)
    
    # Horizontal bar chart
    fig_coef = go.Figure()
    
    colors = ['#2E7D32' if x > 0 else '#DC3545' for x in coef_df['Impact']]
    
    fig_coef.add_trace(go.Bar(
        y=coef_df['Channel'],
        x=coef_df['Impact'],
        orientation='h',
        marker_color=colors,
        text=[f"+{x:.3f}" if x > 0 else f"{x:.3f}" for x in coef_df['Impact']],
        textposition='outside',
        textfont=dict(size=14, weight=700)
    ))
    
    fig_coef.update_layout(
        title="Sales Impact per $1K Advertising Spend",
        xaxis_title="Impact on Sales (K units)",
        height=300,
        margin=dict(t=50, b=50, l=20, r=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans"),
        xaxis=dict(gridcolor='#E4E4E0', zeroline=True, zerolinecolor='#18181B', zerolinewidth=2),
        yaxis=dict(gridcolor='#E4E4E0')
    )
    
    st.plotly_chart(fig_coef, use_container_width=True)
    
    # Insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🥇 Best Channel</div>
            <div class="metric-value">Radio</div>
            <p style="color: #52525B; margin-top: 0.5rem; font-size: 0.9rem;">
                Highest ROI - every $1K generates 0.19K sales
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🥈 Second Best</div>
            <div class="metric-value">TV</div>
            <p style="color: #52525B; margin-top: 0.5rem; font-size: 0.9rem;">
                Good reach - scales well with higher budgets
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">⚠️ Reconsider</div>
            <div class="metric-value">Newspaper</div>
            <p style="color: #52525B; margin-top: 0.5rem; font-size: 0.9rem;">
                Minimal impact - consider reallocating budget
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Historical Marketing Data")
    
    # Scatter plot: TV vs Sales
    fig_scatter = px.scatter(
        df, 
        x='TV', 
        y='Sales',
        color='Radio',
        size='Newspaper',
        color_continuous_scale='RdYlBu_r',
        labels={'TV': 'TV Spend ($K)', 'Sales': 'Sales (K units)', 'Radio': 'Radio ($K)'},
        title="TV Spend vs Sales (sized by Newspaper, colored by Radio)"
    )
    
    fig_scatter.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans"),
        xaxis=dict(gridcolor='#E4E4E0'),
        yaxis=dict(gridcolor='#E4E4E0')
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(df, use_container_width=True, height=300)
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="marketing_data.csv",
            mime="text/csv"
        )

with tab3:
    st.markdown("### Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Algorithm</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #18181B;">Linear Regression</div>
            <p style="color: #52525B; margin-top: 0.5rem; font-size: 0.9rem;">
                Simple, interpretable, and effective for this use case
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Training Data</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #18181B;">210 Records</div>
            <p style="color: #52525B; margin-top: 0.5rem; font-size: 0.9rem;">
                Historical marketing campaigns and sales data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">0.918</div>
            <p style="color: #52525B; margin-top: 0.5rem; font-size: 0.9rem;">
                Model explains 91.8% of sales variance
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">1.44</div>
            <p style="color: #52525B; margin-top: 0.5rem; font-size: 0.9rem;">
                Average prediction error of 1.44K units
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Model Equation:</strong><br>
        <code>Sales = 3.98 + 0.041×TV + 0.192×Radio - 0.012×Newspaper</code>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="footer">
    Built by <a href="https://charlesjovans.netlify.app" target="_blank">Charles Jovans Galiwango</a> · 
    <a href="https://github.com/ChuckJovans" target="_blank">GitHub</a> · 
    <a href="https://linkedin.com/in/charles-jovans-galiwango-2a1194115" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
