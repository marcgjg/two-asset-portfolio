import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Move sliders to sidebar for compact layout
with st.sidebar:
    st.header("Portfolio Parameters")
    er_a = st.slider('Expected Return Stock A (%)', 0.0, 30.0, 12.0, 0.01)
    sigma_a = st.slider('Standard Deviation Stock A (%)', 1.0, 50.0, 15.0, 0.01)
    er_b = st.slider('Expected Return Stock B (%)', 0.0, 30.0, 8.0, 0.01)
    sigma_b = st.slider('Standard Deviation Stock B (%)', 1.0, 50.0, 20.0, 0.01)
    rho = st.slider('Correlation Coefficient (ρ)', -1.0, 1.0, 0.3, 0.01)

# Convert to decimals
er_a, er_b = er_a/100, er_b/100
sigma_a, sigma_b = sigma_a/100, sigma_b/100
cov = rho * sigma_a * sigma_b

# Calculate MVP weights using general formula
numerator = sigma_b**2 - rho*sigma_a*sigma_b
denominator = sigma_a**2 + sigma_b**2 - 2*rho*sigma_a*sigma_b
w_mvp = numerator / denominator if denominator != 0 else 0.5

# MVP coordinates
mvp_return = w_mvp*er_a + (1-w_mvp)*er_b
mvp_vol = np.sqrt(w_mvp**2 * sigma_a**2 + (1-w_mvp)**2 * sigma_b**2 + 2*w_mvp*(1-w_mvp)*cov)

fig, ax = plt.subplots(figsize=(10, 5))

if np.isclose(er_a, er_b):
    # Equal returns case
    ax.scatter(mvp_vol*100, mvp_return*100, color='red', s=100, label='MVP (Efficient Frontier)')
    ax.scatter(mvp_vol*100, mvp_return*100, marker='*', s=400, edgecolor='black', facecolor='none')
else:
    # Different returns case
    weights = np.linspace(-0.5, 1.5, 1000)
    returns = weights*er_a + (1-weights)*er_b
    volatilities = np.sqrt(weights**2 * sigma_a**2 + (1-weights)**2 * sigma_b**2 + 2*weights*(1-weights)*cov)
    
    # Split curve at MVP
    idx_mvp = np.argmin(volatilities)
    efficient_mask = weights >= w_mvp if er_a > er_b else weights <= w_mvp
    
    ax.plot(volatilities[efficient_mask]*100, returns[efficient_mask]*100, 'r-', label='Efficient Frontier')
    ax.plot(volatilities[~efficient_mask]*100, returns[~efficient_mask]*100, 'r--', label='Inefficient Frontier')
    ax.scatter(mvp_vol*100, mvp_return*100, color='red', s=100, label='MVP')
    ax.scatter(mvp_vol*100, mvp_return*100, marker='*', s=400, edgecolor='black', facecolor='none')
    
    # Add random portfolios
    if st.sidebar.checkbox('Show Random Portfolios'):
        random_weights = np.random.uniform(-0.5, 1.5, 1000)
        random_returns = random_weights*er_a + (1-random_weights)*er_b
        random_vols = np.sqrt(random_weights**2 * sigma_a**2 + (1-random_weights)**2 * sigma_b**2 + 2*random_weights*(1-random_weights)*cov)
        ax.scatter(random_vols*100, random_returns*100, color='gray', alpha=0.3, s=10)

# Plot individual stocks
ax.scatter(sigma_a*100, er_a*100, color='blue', s=100, label='Stock A', marker='o')
ax.scatter(sigma_b*100, er_b*100, color='green', s=100, label='Stock B', marker='s')

# Formatting
ax.set_xlabel('Volatility (%)', fontweight='bold')
ax.set_ylabel('Return (%)', fontweight='bold')
ax.set_title(f"Efficient Frontier | MVP at ({mvp_vol*100:.1f}%, {mvp_return*100:.1f}%)\n"
             f"Stock A: ({sigma_a*100:.1f}%, {er_a*100:.1f}%) | "
             f"Stock B: ({sigma_b*100:.1f}%, {er_b*100:.1f}%)", 
             pad=20, fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Display plot without scrolling
st.pyplot(fig)
