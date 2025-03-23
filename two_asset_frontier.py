import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Create input columns
col1, col2 = st.columns(2)
with col1:
    er_a = st.slider('Expected Return Stock A (%)', 0.0, 30.0, 10.0, 0.01)
    sigma_a = st.slider('Standard Deviation Stock A (%)', 1.0, 50.0, 15.0, 0.01)
with col2:
    er_b = st.slider('Expected Return Stock B (%)', 0.0, 30.0, 10.0, 0.01)
    sigma_b = st.slider('Standard Deviation Stock B (%)', 1.0, 50.0, 20.0, 0.01)
rho = st.slider('Correlation Coefficient (ρ)', -1.0, 1.0, 0.5, 0.01)

# Convert percentages to decimals
er_a /= 100
er_b /= 100
sigma_a /= 100
sigma_b /= 100

# Calculate covariance
cov = rho * sigma_a * sigma_b

fig, ax = plt.subplots(figsize=(10, 6))

if np.isclose(er_a, er_b):
    # Case 1: Same expected returns
    numerator = sigma_b**2 - rho*sigma_a*sigma_b
    denominator = sigma_a**2 + sigma_b**2 - 2*rho*sigma_a*sigma_b
    w_a = numerator / denominator
    w_b = 1 - w_a
    
    port_return = er_a  # Since er_a = er_b
    port_vol = np.sqrt(w_a**2 * sigma_a**2 + w_b**2 * sigma_b**2 + 2*w_a*w_b*cov)
    
    ax.scatter(port_vol*100, port_return*100, color='red', s=100, label='MVP')
    ax.scatter(port_vol*100, port_return*100, marker='*', s=400, edgecolor='black', facecolor='none')
    
else:
    # Case 2: Different expected returns
    weights = np.linspace(-0.5, 1.5, 100)
    returns = weights*er_a + (1-weights)*er_b
    volatilities = np.sqrt(weights**2 * sigma_a**2 + (1-weights)**2 * sigma_b**2 + 2*weights*(1-weights)*cov)
    
    # Split into efficient and inefficient frontiers
    efficient_mask = returns >= min(er_a, er_b)
    ax.plot(volatilities[efficient_mask]*100, returns[efficient_mask]*100, 'r-', label='Efficient Frontier')
    ax.plot(volatilities[~efficient_mask]*100, returns[~efficient_mask]*100, 'r--', label='Inefficient')
    
    # Optional random portfolios
    if st.checkbox('Show Random Portfolios'):
        random_weights = np.random.uniform(-0.5, 1.5, 1000)
        random_returns = random_weights*er_a + (1-random_weights)*er_b
        random_vols = np.sqrt(random_weights**2 * sigma_a**2 + (1-random_weights)**2 * sigma_b**2 + 2*random_weights*(1-random_weights)*cov)
        ax.scatter(random_vols*100, random_returns*100, color='gray', alpha=0.3, s=10)

# Plot formatting
ax.set_xlabel('Portfolio Volatility (%)')
ax.set_ylabel('Portfolio Return (%)')
ax.set_title('Two-Asset Efficient Frontier')
ax.grid(True)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)
