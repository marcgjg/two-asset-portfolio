import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set layout to wide
st.set_page_config(layout="wide")

# Define sliders for inputs
col1, col2, col3 = st.columns(3)
with col1:
    mu_A = st.slider('Expected Return of Stock A', min_value=0.0, max_value=1.0, value=0.05, step=0.01)
with col2:
    mu_B = st.slider('Expected Return of Stock B', min_value=0.0, max_value=1.0, value=0.05, step=0.01)
with col3:
    sigma_A = st.slider('Standard Deviation of Stock A', min_value=0.0, max_value=1.0, value=0.1, step=0.01)

col4, col5 = st.columns(2)
with col4:
    sigma_B = st.slider('Standard Deviation of Stock B', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
with col5:
    rho = st.slider('Correlation Coefficient', min_value=-1.0, max_value=1.0, value=0.5, step=0.01)

# Compute Minimum Variance Portfolio if returns are equal
if mu_A == mu_B:
    w_star = (sigma_B**2 - rho * sigma_A * sigma_B) / (sigma_A**2 + sigma_B**2 - 2 * rho * sigma_A * sigma_B)
    portfolio_return = mu_A
    portfolio_std = np.sqrt(w_star**2 * sigma_A**2 + (1 - w_star)**2 * sigma_B**2 + 2 * w_star * (1 - w_star) * rho * sigma_A * sigma_B)

    # Plot MVP
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(portfolio_std, portfolio_return, color='red', label='Efficient Frontier')
    ax.scatter(portfolio_std, portfolio_return, marker='*', color='black')
    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Expected Return')
    ax.set_title('Minimum Variance Portfolio')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    st.pyplot(fig)

else:
    # Generate parametric efficient frontier
    alphas = np.linspace(0, 1, 100)
    portfolio_returns = alphas * mu_A + (1 - alphas) * mu_B
    portfolio_stds = np.sqrt(alphas**2 * sigma_A**2 + (1 - alphas)**2 * sigma_B**2 + 2 * alphas * (1 - alphas) * rho * sigma_A * sigma_B)

    # Split into efficient and inefficient parts
    max_return_idx = np.argmax(portfolio_returns)
    efficient_returns = portfolio_returns[:max_return_idx+1]
    efficient_stds = portfolio_stds[:max_return_idx+1]
    inefficient_returns = portfolio_returns[max_return_idx:]
    inefficient_stds = portfolio_stds[max_return_idx:]

    # Plot efficient frontier
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(efficient_stds, efficient_returns, color='red', label='Efficient Frontier')
    ax.plot(inefficient_stds, inefficient_returns, color='red', linestyle='--', label='Inefficient Frontier')

    # Optionally include random portfolios
    if st.checkbox('Include Random Portfolios'):
        random_alphas = np.random.uniform(0, 1, size=100)
        random_returns = random_alphas * mu_A + (1 - random_alphas) * mu_B
        random_stds = np.sqrt(random_alphas**2 * sigma_A**2 + (1 - random_alphas)**2 * sigma_B**2 + 2 * random_alphas * (1 - random_alphas) * rho * sigma_A * sigma_B)
        ax.scatter(random_stds, random_returns, color='gray', alpha=0.5)

    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    st.pyplot(fig)
