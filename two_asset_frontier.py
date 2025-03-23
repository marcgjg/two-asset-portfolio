import plotly.express as px
import numpy as np
import streamlit as st

# Set layout to wide
st.set_page_config(layout="wide")

# Define sliders for inputs
col1, col2 = st.columns(2)
with col1:
    mu_A = st.slider('Expected Return of Stock A (%)', min_value=0.0, max_value=50.0, value=7.90, step=0.1)
with col2:
    mu_B = st.slider('Expected Return of Stock B (%)', min_value=0.0, max_value=50.0, value=8.20, step=0.1)

col3, col4 = st.columns(2)
with col3:
    sigma_A = st.slider('Standard Deviation of Stock A (%)', min_value=0.0, max_value=50.0, value=10.00, step=0.1)
with col4:
    sigma_B = st.slider('Standard Deviation of Stock B (%)', min_value=0.0, max_value=50.0, value=13.90, step=0.1)

rho = st.slider('Correlation Coefficient', min_value=-1.0, max_value=1.0, value=0.74, step=0.01)

# Convert sliders back to decimal form for calculations
mu_A /= 100
mu_B /= 100
sigma_A /= 100
sigma_B /= 100

# Generate parametric minimum-variance frontier
alphas = np.linspace(0, 1, 100)
portfolio_returns = alphas * mu_A + (1 - alphas) * mu_B
portfolio_stds = np.sqrt(
    alphas**2 * sigma_A**2 +
    (1 - alphas)**2 * sigma_B**2 +
    2 * alphas * (1 - alphas) * rho * sigma_A * sigma_B
)

# Compute Minimum Variance Portfolio (MVP)
w_star = (sigma_B**2 - rho * sigma_A * sigma_B) / (sigma_A**2 + sigma_B**2 - 2 * rho * sigma_A * sigma_B)
w_star = max(0, min(w_star, 1))  # Ensure no short sales
mvp_return = w_star * mu_A + (1 - w_star) * mu_B
mvp_std = np.sqrt(
    w_star**2 * sigma_A**2 +
    (1 - w_star)**2 * sigma_B**2 +
    2 * w_star * (1 - w_star) * rho * sigma_A * sigma_B
)

# Special case handling: If returns are equal, MVP is the only efficient portfolio
if mu_A == mu_B:
    # Plot MVP as a single point
    fig = px.scatter(x=[sigma_A * 100], y=[mu_A * 100], text=['Stock A'])
    fig.add_scatter(x=[sigma_B * 100], y=[mu_B * 100], text=['Stock B'])
    fig.add_scatter(x=[mvp_std * 100], y=[mvp_return * 100], text=['MVP'], mode='markers', marker=dict(size=20))
    fig.update_layout(
        title='Minimum Variance Portfolio',
        xaxis_title='Standard Deviation (%)',
        yaxis_title='Expected Return (%)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)  # Adjust legend position
    )
else:
    # Split into efficient and inefficient frontiers
    efficient_mask = portfolio_returns >= mvp_return  # Keep only points above or equal to MVP's return
    efficient_returns = portfolio_returns[efficient_mask]
    efficient_stds = portfolio_stds[efficient_mask]
    inefficient_returns = portfolio_returns[~efficient_mask]
    inefficient_stds = portfolio_stds[~efficient_mask]

    # Plotting
    fig = px.scatter(x=[sigma_A * 100], y=[mu_A * 100], text=['Stock A'])
    fig.add_scatter(x=[sigma_B * 100], y=[mu_B * 100], text=['Stock B'])
    fig.add_scatter(x=[mvp_std * 100], y=[mvp_return * 100], text=['MVP'], mode='markers', marker=dict(size=20))
    fig.add_scatter(x=efficient_stds * 100, y=efficient_returns * 100, mode='lines', line=dict(color='red'), name='Efficient Frontier')
    fig.add_scatter(x=inefficient_stds * 100, y=inefficient_returns * 100, mode='lines', line=dict(color='red', dash='dash'), name='Inefficient Frontier')
    fig.update_layout(
        title='Efficient Frontier with MVP',
        xaxis_title='Standard Deviation (%)',
        yaxis_title='Expected Return (%)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)  # Adjust legend position
    )

# Display plot in Streamlit app
st.plotly_chart(fig, use_container_width=True)
