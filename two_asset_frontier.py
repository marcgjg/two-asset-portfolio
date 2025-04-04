import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set layout to wide
st.set_page_config(layout="wide")

# Create columns for sliders and plot
col1, col2 = st.columns([2, 6])

# Define sliders for inputs
with col1:
    mu_A = st.slider('Expected Return of Stock A (%)', min_value=0.0, max_value=50.0, value=8.90, step=0.1)
    mu_B = st.slider('Expected Return of Stock B (%)', min_value=0.0, max_value=50.0, value=9.20, step=0.1)
    sigma_A = st.slider('Standard Deviation of Stock A (%)', min_value=0.0, max_value=50.0, value=7.90, step=0.1)
    sigma_B = st.slider('Standard Deviation of Stock B (%)', min_value=0.0, max_value=50.0, value=8.90, step=0.1)
    rho = st.slider('Correlation Coefficient', min_value=-1.0, max_value=1.0, value=-1.0, step=0.01)

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
denominator = sigma_A**2 + sigma_B**2 - 2 * rho * sigma_A * sigma_B

# Handle division by zero
if denominator == 0:
    w_star = sigma_B / (sigma_A + sigma_B)  # Special handling for rho = -1
else:
    w_star = (sigma_B**2 - rho * sigma_A * sigma_B) / denominator

w_star = max(0, min(w_star, 1))  # Ensure no short sales

mvp_return = w_star * mu_A + (1 - w_star) * mu_B

# Correctly calculate MVP standard deviation
mvp_variance = w_star**2 * sigma_A**2 + (1 - w_star)**2 * sigma_B**2 + 2 * w_star * (1 - w_star) * rho * sigma_A * sigma_B

# Check if variance is non-negative before calculating standard deviation
if mvp_variance >= 0:
    mvp_std = np.sqrt(mvp_variance)
else:
    mvp_std = 0  # Set to 0 if variance is negative (theoretical minimum variance portfolio)

# Special case handling: If returns are equal, MVP is the only efficient portfolio
if mu_A == mu_B:
    # Plot MVP as a single point
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(sigma_A * 100, mu_A * 100, color='blue', label='Stock A')
    ax.scatter(sigma_B * 100, mu_B * 100, color='green', label='Stock B')
    ax.scatter(mvp_std * 100, mvp_return * 100, marker='*', color='black', s=200, label=f'MVP ({mvp_std*100:.2f}, {mvp_return*100:.2f})')
    ax.set_xlabel('Standard Deviation (%)')
    ax.set_ylabel('Expected Return (%)')
    ax.set_xlim(0, max(sigma_A, sigma_B) * 100 + 1)  # Set x-axis limits
    ax.set_ylim(min(mu_A, mu_B) * 100 - 1, max(mu_A, mu_B) * 100 + 1)  # Set y-axis limits
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title('Minimum Variance Portfolio')
else:
    # Split into efficient and inefficient frontiers
    efficient_mask = portfolio_returns >= mvp_return  # Keep only points above or equal to MVP's return
    efficient_returns = portfolio_returns[efficient_mask]
    efficient_stds = portfolio_stds[efficient_mask]
    inefficient_returns = portfolio_returns[~efficient_mask]
    inefficient_stds = portfolio_stds[~efficient_mask]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(sigma_A * 100, mu_A * 100, color='blue', label='Stock A')
    ax.scatter(sigma_B * 100, mu_B * 100, color='green', label='Stock B')
    ax.scatter(mvp_std * 100, mvp_return * 100, marker='*', color='black', s=200, label=f'MVP ({mvp_std*100:.2f}, {mvp_return*100:.2f})')
    ax.plot(efficient_stds * 100, efficient_returns * 100, color='red', label='Efficient Frontier')
    ax.plot(inefficient_stds * 100, inefficient_returns * 100, color='red', linestyle='--', label='Inefficient Frontier')
    
    # Set x-axis limits with default value for empty lists
    max_efficient_std = max(efficient_stds, default=0)
    max_inefficient_std = max(inefficient_stds, default=0)
    ax.set_xlabel('Standard Deviation (%)')
    ax.set_ylabel('Expected Return (%)')
    ax.set_xlim(0, max(max_efficient_std, max_inefficient_std) * 100 + 1)  # Set x-axis limits
    ax.set_ylim(min(min(efficient_returns), min(inefficient_returns, default=mu_A)) * 100 - 1, max(max(efficient_returns), max(inefficient_returns, default=mu_B)) * 100 + 1)  # Set y-axis limits
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title('Efficient Frontier with Minimum-Variance Portfolio (MVP)')

# Display plot in the second column
with col2:
    st.pyplot(fig)
