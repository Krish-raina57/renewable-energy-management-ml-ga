"""
Renewable Energy Management — Streamlit Dashboard
===================================================
Interactive web app for exploring PV generation, electricity demand,
ML forecasting models, and energy optimization strategies (GA & MILP).

Author: Krish Raina
GitHub: https://github.com/Krish-raina57/renewable-energy-management-ml-ga
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="🔋 Renewable Energy Management",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom Styling
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2E7D32, #1565C0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.markdown('<p class="main-header">🔋 Renewable Energy Management System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-Powered PV Forecasting · Demand Modeling · GA & MILP Optimization</p>', unsafe_allow_html=True)

# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data
def load_pv_data():
    """Load PV generation dataset."""
    try:
        df = pd.read_csv('data/PV_dataset_clean.csv')
    except FileNotFoundError:
        df = _generate_synthetic_pv()
    return df

@st.cache_data
def load_demand_data():
    """Load electricity demand dataset."""
    try:
        df = pd.read_csv('data/Clean_demand_dataset.csv')
    except FileNotFoundError:
        df = _generate_synthetic_demand()
    return df

def _generate_synthetic_pv():
    """Fallback: generate synthetic PV data matching the real dataset structure."""
    np.random.seed(42)
    rows = []
    for month in range(1, 13):
        for day in range(1, 29):
            for hour in range(6, 20):
                sun_height = max(0, 30 * np.sin(np.pi * (hour - 6) / 14) + np.random.normal(0, 3))
                direct = max(0, sun_height * 3.5 + np.random.normal(0, 15))
                seasonal = 1.0 + 0.4 * np.sin(2 * np.pi * (month - 3) / 12)
                pv = max(0, direct * 0.7 * seasonal + np.random.normal(0, 10))
                temp = 10 + 15 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3)
                wind = max(0, 4 + np.random.normal(0, 2))
                rows.append([pv, direct, sun_height, temp, wind, month, day, hour])
    return pd.DataFrame(rows, columns=['PV power', 'Direct-in-plane', 'Sun Height',
                                        'Air Temperature', 'Wind speed', 'month', 'day', 'hour_minute'])

def _generate_synthetic_demand():
    """Fallback: generate synthetic demand data matching the real dataset structure."""
    np.random.seed(42)
    rows = []
    for month in range(1, 13):
        for day in range(1, 29):
            for hour in range(1, 25):
                seasonal = 1.0 + 0.3 * np.cos(2 * np.pi * (month - 1) / 12)
                hourly_base = np.array([80, 60, 50, 45, 50, 70, 120, 180, 200, 180,
                                        160, 150, 140, 130, 140, 160, 200, 250, 280,
                                        260, 220, 180, 140, 100])
                consumption = hourly_base[hour-1] * seasonal + np.random.normal(0, 20)
                production = max(0, 150 * np.sin(np.pi * max(0, hour - 6) / 14) * seasonal) if 6 <= hour <= 20 else 0
                from_grid = max(0, consumption - production + np.random.normal(0, 10))
                feed_in = max(0, production - consumption) if production > consumption else 0
                soc = max(0, min(100, 50 + 30 * np.sin(2 * np.pi * (hour - 10) / 24) + np.random.normal(0, 5)))
                charge = max(0, production * 0.1 + np.random.normal(0, 5))
                discharge = max(0, consumption * 0.05 + np.random.normal(0, 3))
                rows.append([discharge, charge, production, consumption, feed_in,
                             from_grid, soc, month, day, hour])
    return pd.DataFrame(rows, columns=['Discharge(Wh)', 'Charge(Wh)', 'Production(Wh)',
                                        'Consumption(Wh)', 'Feed-in(Wh)', 'From grid(Wh)',
                                        'State of Charge(%)', 'Month', 'Day', 'Hour'])

pv_df = load_pv_data()
demand_df = load_demand_data()

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.header("🎛️ Dashboard Controls")

month_filter = st.sidebar.multiselect(
    "Filter by Month",
    options=list(range(1, 13)),
    default=list(range(1, 13)),
    format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1]
)

pv_filtered = pv_df[pv_df['month'].isin(month_filter)]

# Map demand month column
demand_month_col = 'Month' if 'Month' in demand_df.columns else 'month'
demand_hour_col = 'Hour' if 'Hour' in demand_df.columns else 'hour'
demand_filtered = demand_df[demand_df[demand_month_col].isin(month_filter)]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Stats")
st.sidebar.markdown(f"**PV Dataset:** {len(pv_df):,} records × {pv_df.shape[1]} columns")
st.sidebar.markdown(f"**Demand Dataset:** {len(demand_df):,} records × {demand_df.shape[1]} columns")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Research Paper")
st.sidebar.markdown("[Showole & Afaqui (2026)](https://doi.org/10.1016/j.enss.2025.06.007)")
st.sidebar.markdown("*Energy Storage and Saving*")

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "☀️ PV Generation", "⚡ Demand Analysis", "🤖 ML Models",
    "🔧 Optimization", "📊 System Overview"
])

# =============================================================================
# Tab 1: PV Generation Analysis
# =============================================================================
with tab1:
    st.markdown("### ☀️ Solar PV Generation Analysis")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("☀️ Avg PV Power", f"{pv_filtered['PV power'].mean():.1f} W")
    col2.metric("🔆 Peak PV Power", f"{pv_filtered['PV power'].max():.1f} W")
    col3.metric("🌡️ Avg Temperature", f"{pv_filtered['Air Temperature'].mean():.1f} °C")
    col4.metric("💨 Avg Wind Speed", f"{pv_filtered['Wind speed'].mean():.1f} m/s")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Average PV Power by Hour")
        hourly_pv = pv_filtered.groupby('hour_minute')['PV power'].mean().reset_index()
        fig = px.area(hourly_pv, x='hour_minute', y='PV power',
                      labels={'hour_minute': 'Hour of Day', 'PV power': 'Avg PV Power (W)'},
                      color_discrete_sequence=['#FFA726'])
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Monthly PV Generation")
        monthly_pv = pv_filtered.groupby('month')['PV power'].agg(['mean', 'max']).reset_index()
        monthly_pv.columns = ['Month', 'Average', 'Peak']
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pv['Month_Name'] = monthly_pv['Month'].apply(lambda x: month_names[int(x)-1])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_pv['Month_Name'], y=monthly_pv['Average'],
                             name='Average', marker_color='#FFA726'))
        fig.add_trace(go.Bar(x=monthly_pv['Month_Name'], y=monthly_pv['Peak'],
                             name='Peak', marker_color='#FF7043'))
        fig.update_layout(barmode='group', template='plotly_white', height=400,
                          yaxis_title='PV Power (W)')
        st.plotly_chart(fig, use_container_width=True)

    # Feature Correlation
    st.markdown("#### PV Feature Correlation Heatmap")
    numeric_cols = ['PV power', 'Direct-in-plane', 'Sun Height', 'Air Temperature', 'Wind speed']
    available_cols = [c for c in numeric_cols if c in pv_filtered.columns]
    corr = pv_filtered[available_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdYlGn',
                    labels=dict(color="Correlation"), aspect='auto')
    fig.update_layout(template='plotly_white', height=450)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Tab 2: Demand Analysis
# =============================================================================
with tab2:
    st.markdown("### ⚡ Electricity Demand Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔌 Avg Consumption", f"{demand_filtered['Consumption(Wh)'].mean():.0f} Wh")
    col2.metric("📈 Peak Consumption", f"{demand_filtered['Consumption(Wh)'].max():.0f} Wh")
    col3.metric("🏭 Avg Production", f"{demand_filtered['Production(Wh)'].mean():.0f} Wh")
    col4.metric("🔋 Avg SoC", f"{demand_filtered['State of Charge(%)'].mean():.1f}%")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Hourly Energy Profile")
        hourly_demand = demand_filtered.groupby(demand_hour_col).agg({
            'Consumption(Wh)': 'mean',
            'Production(Wh)': 'mean',
            'From grid(Wh)': 'mean'
        }).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_demand[demand_hour_col], y=hourly_demand['Consumption(Wh)'],
                                 mode='lines+markers', name='Consumption',
                                 line=dict(color='#EF5350', width=3)))
        fig.add_trace(go.Scatter(x=hourly_demand[demand_hour_col], y=hourly_demand['Production(Wh)'],
                                 mode='lines+markers', name='Production',
                                 line=dict(color='#66BB6A', width=3)))
        fig.add_trace(go.Scatter(x=hourly_demand[demand_hour_col], y=hourly_demand['From grid(Wh)'],
                                 mode='lines+markers', name='From Grid',
                                 line=dict(color='#42A5F5', width=3)))
        fig.update_layout(template='plotly_white', height=400,
                          xaxis_title='Hour of Day', yaxis_title='Energy (Wh)',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Battery State of Charge")
        hourly_soc = demand_filtered.groupby(demand_hour_col)['State of Charge(%)'].mean().reset_index()
        fig = px.area(hourly_soc, x=demand_hour_col, y='State of Charge(%)',
                      labels={demand_hour_col: 'Hour of Day'},
                      color_discrete_sequence=['#AB47BC'])
        fig.update_layout(template='plotly_white', height=400,
                          yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

    # Monthly breakdown
    st.markdown("#### Monthly Energy Balance")
    monthly_demand = demand_filtered.groupby(demand_month_col).agg({
        'Consumption(Wh)': 'sum',
        'Production(Wh)': 'sum',
        'From grid(Wh)': 'sum',
        'Feed-in(Wh)': 'sum'
    }).reset_index()
    monthly_demand['Month_Name'] = monthly_demand[demand_month_col].apply(
        lambda x: month_names[int(x)-1] if int(x) <= 12 else str(x))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_demand['Month_Name'], y=monthly_demand['Consumption(Wh)'],
                         name='Consumption', marker_color='#EF5350'))
    fig.add_trace(go.Bar(x=monthly_demand['Month_Name'], y=monthly_demand['Production(Wh)'],
                         name='Production', marker_color='#66BB6A'))
    fig.add_trace(go.Bar(x=monthly_demand['Month_Name'], y=monthly_demand['From grid(Wh)'],
                         name='From Grid', marker_color='#42A5F5'))
    fig.update_layout(barmode='group', template='plotly_white', height=400,
                      yaxis_title='Energy (Wh)')
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Tab 3: ML Model Comparison
# =============================================================================
with tab3:
    st.markdown("### 🤖 Machine Learning Model Performance")

    col_pv, col_demand = st.columns(2)

    with col_pv:
        st.markdown("#### ☀️ PV Forecasting Models")
        pv_models = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Stacking Ensemble'],
            'R²': [0.9994, 0.9996, 0.9995],
            'MAE': [3.1659, 2.6737, 2.8943]
        })

        fig = make_subplots(rows=1, cols=2, subplot_titles=('R² Score', 'MAE (Lower is Better)'))

        colors_pv = ['#4CAF50', '#FF9800', '#2196F3']
        fig.add_trace(go.Bar(x=pv_models['Model'], y=pv_models['R²'],
                             marker_color=colors_pv, text=pv_models['R²'].round(4),
                             textposition='outside', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=pv_models['Model'], y=pv_models['MAE'],
                             marker_color=colors_pv, text=pv_models['MAE'].round(4),
                             textposition='outside', showlegend=False), row=1, col=2)

        fig.update_layout(template='plotly_white', height=400)
        fig.update_yaxes(range=[0.998, 1.0], row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

        st.success("🏆 **Best PV Model:** XGBoost (lowest MAE: 2.6737)")

    with col_demand:
        st.markdown("#### ⚡ Demand Forecasting Models")
        demand_models = pd.DataFrame({
            'Model': ['Linear Regression', 'Decision Tree', 'Gradient Boosting', 'Stacking Ensemble'],
            'R²': [0.9998, 0.9810, 0.9969, 0.9999],
            'MAE': [2.7961, 35.5813, 22.6532, 2.8049]
        })

        fig = make_subplots(rows=1, cols=2, subplot_titles=('R² Score', 'MAE (Lower is Better)'))

        colors_demand = ['#9C27B0', '#E91E63', '#FF5722', '#3F51B5']
        fig.add_trace(go.Bar(x=demand_models['Model'], y=demand_models['R²'],
                             marker_color=colors_demand, text=demand_models['R²'].round(4),
                             textposition='outside', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=demand_models['Model'], y=demand_models['MAE'],
                             marker_color=colors_demand, text=demand_models['MAE'].round(4),
                             textposition='outside', showlegend=False), row=1, col=2)

        fig.update_layout(template='plotly_white', height=400)
        fig.update_yaxes(range=[0.97, 1.0], row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

        st.success("🏆 **Best Demand Model:** Stacking Ensemble (highest R²: 0.9999)")

    # Combined comparison table
    st.markdown("---")
    st.markdown("#### 📋 Complete Model Comparison")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**PV Forecasting**")
        st.dataframe(pv_models.style.highlight_min(subset=['MAE'], color='#C8E6C9')
                     .highlight_max(subset=['R²'], color='#C8E6C9')
                     .format({'R²': '{:.4f}', 'MAE': '{:.4f}'}),
                     use_container_width=True, hide_index=True)
    with col_t2:
        st.markdown("**Demand Forecasting**")
        st.dataframe(demand_models.style.highlight_min(subset=['MAE'], color='#C8E6C9')
                     .highlight_max(subset=['R²'], color='#C8E6C9')
                     .format({'R²': '{:.4f}', 'MAE': '{:.4f}'}),
                     use_container_width=True, hide_index=True)

# =============================================================================
# Tab 4: Optimization Strategies
# =============================================================================
with tab4:
    st.markdown("### 🔧 Energy Management Optimization")

    # Optimization results
    opt_data = pd.DataFrame({
        'Strategy': ['Standard GA', 'Hybrid GA V2', 'MILP'],
        'Grid Reduction (%)': [3.69, 2.23, 3.69],
        'Cost Reduction (%)': [6.04, 5.98, 7.54]
    })

    col1, col2, col3 = st.columns(3)
    col1.metric("🔌 Best Grid Reduction", "3.69%", "Standard GA & MILP")
    col2.metric("💰 Best Cost Reduction", "7.54%", "MILP")
    col3.metric("🏆 Overall Best", "MILP", "Highest cost savings")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Grid Reduction Comparison")
        fig = px.bar(opt_data, x='Strategy', y='Grid Reduction (%)',
                     color='Strategy', text='Grid Reduction (%)',
                     color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3'])
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(template='plotly_white', height=400, showlegend=False,
                          yaxis=dict(range=[0, 5]))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Cost Reduction Comparison")
        fig = px.bar(opt_data, x='Strategy', y='Cost Reduction (%)',
                     color='Strategy', text='Cost Reduction (%)',
                     color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3'])
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(template='plotly_white', height=400, showlegend=False,
                          yaxis=dict(range=[0, 10]))
        st.plotly_chart(fig, use_container_width=True)

    # Interactive cost savings calculator
    st.markdown("---")
    st.markdown("#### 💰 Cost Savings Calculator")
    st.markdown("Estimate annual savings based on your electricity costs:")

    calc_col1, calc_col2, calc_col3 = st.columns(3)
    annual_bill = calc_col1.number_input("Annual Electricity Bill (₹)", value=120000, step=5000)
    strategy = calc_col2.selectbox("Optimization Strategy", opt_data['Strategy'].tolist(), index=2)
    grid_pct = opt_data[opt_data['Strategy'] == strategy]['Grid Reduction (%)'].values[0]
    cost_pct = opt_data[opt_data['Strategy'] == strategy]['Cost Reduction (%)'].values[0]

    cost_savings = annual_bill * (cost_pct / 100)
    grid_savings_kwh = (annual_bill / 8) * (grid_pct / 100)  # Approximate kWh at ₹8/kWh

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("💰 Estimated Annual Savings", f"₹{cost_savings:,.0f}")
    res_col2.metric("📉 Cost Reduction", f"{cost_pct}%")
    res_col3.metric("🔌 Grid Dependency Reduction", f"{grid_pct}%")

    st.info(f"💡 Using **{strategy}**, you could save approximately **₹{cost_savings:,.0f}/year** "
            f"on an annual bill of ₹{annual_bill:,}.")

# =============================================================================
# Tab 5: System Overview
# =============================================================================
with tab5:
    st.markdown("### 📊 System Architecture & Overview")

    # Architecture diagram using Mermaid-style flow (rendered as text)
    st.markdown("#### System Pipeline")

    pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = st.columns(4)
    with pipeline_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFA726, #FF7043); padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;">
            <h3>📥 Data</h3>
            <p>PV: 4,284 rows</p>
            <p>Demand: 8,784 rows</p>
        </div>
        """, unsafe_allow_html=True)
    with pipeline_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #66BB6A, #43A047); padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;">
            <h3>🤖 ML Models</h3>
            <p>RF, XGBoost, Stacking</p>
            <p>R² > 0.99</p>
        </div>
        """, unsafe_allow_html=True)
    with pipeline_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #42A5F5, #1E88E5); padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;">
            <h3>⚡ Profiles</h3>
            <p>24-hour energy profiles</p>
            <p>PV + Demand</p>
        </div>
        """, unsafe_allow_html=True)
    with pipeline_col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #AB47BC, #8E24AA); padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;">
            <h3>🔧 Optimize</h3>
            <p>GA & MILP</p>
            <p>Up to 7.54% savings</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Key highlights
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔑 Key Highlights")
        st.markdown("""
        - 🔬 **Multi-model comparison** across PV and demand forecasting
        - 🧬 **Genetic Algorithm** optimization with standard and hybrid variants
        - 📐 **MILP** (Mixed Integer Linear Programming) as optimization baseline
        - 📊 **Real-world data** from PVGIS and Irish energy community
        - 📄 **Published research** in *Energy Storage and Saving* (2026)
        """)

    with col_right:
        st.markdown("#### 🛠️ Tech Stack")
        tech_data = {
            'Category': ['ML Framework', 'ML Framework', 'Optimization', 'Optimization', 'Data', 'Visualization'],
            'Tool': ['scikit-learn', 'XGBoost', 'Genetic Algorithm', 'PuLP (MILP)', 'Pandas / NumPy', 'Plotly / Streamlit'],
            'Purpose': ['RF, Stacking, Metrics', 'Gradient Boosting', 'Battery scheduling', 'Exact optimization', 'Data processing', 'Interactive dashboard']
        }
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    # Data explorer
    st.markdown("---")
    st.markdown("#### 🔍 Data Explorer")
    dataset_choice = st.selectbox("Select Dataset", ["PV Dataset", "Demand Dataset"])
    if dataset_choice == "PV Dataset":
        st.dataframe(pv_filtered.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(pv_filtered):,} filtered records")
    else:
        st.dataframe(demand_filtered.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(demand_filtered):,} filtered records")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    Built by <strong>Krish Raina</strong> · B.Tech CSE, SMVDU ·
    <a href="https://github.com/Krish-raina57/renewable-energy-management-ml-ga" target="_blank">GitHub Repo</a> ·
    <a href="https://www.linkedin.com/in/krish-raina" target="_blank">LinkedIn</a><br/>
    <em>Based on: Showole & Afaqui, "Optimization of renewable energy management using ML and GA,"
    Energy Storage and Saving (2026)</em>
</div>
""", unsafe_allow_html=True)
