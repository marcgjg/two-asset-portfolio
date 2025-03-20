import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1) Use a wide page layout to maximize horizontal space
st.set_page_config(layout='wide')

def plot_two_asset_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    # Compute covariance
    cov_AB = corr_AB * sigma_A * sigma_B

    # Generate parametric frontier
    weights = np.linspace(0, 1, 100)
    port_returns = []
    port_stdevs  = []

    for w in weights:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        port_returns.append(p_return)
        port_stdevs.append(np.sqrt(p_var))

    # Random simulation (optional, just for visual illustration)
    n_portfolios = 3000
    rand_w = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []

    for w in rand_w:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(p_return)
        rand_stdevs.append(np.sqrt(p_var))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))  # Reduced figure size
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, label='Random Portfolios')
    ax.plot(port_stdevs, port_returns, 'r-', label='Efficient Frontier', linewidth=2)
    
    # Mark individual assets
    ax.scatter(sigma_A, mu_A, s=100, marker='o', label='Asset A')
    ax.scatter(sigma_B, mu_B, s=100, marker='o', label='Asset B')

    ax.set_title('2-Stock Efficient Frontier')
    ax.set_xlabel('Portfolio Standard Deviation')
    ax.set_ylabel('Portfolio Expected Return')
    ax.legend()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Efficient Frontier Simulator")

    # 2) Create two columns: left for sliders, right for chart
    col1, col2 = st.columns([1,2])  # 1:2 width ratio

    with col1:
        st.markdown("### Adjust Parameters")
        mu_A = st.slider("mu_A (Expected Return of Asset A)", 0.00, 0.20, 0.10, 0.01)
        mu_B = st.slider("mu_B (Expected Return of Asset B)", 0.00, 0.20, 0.15, 0.01)
        sigma_A = st.slider("sigma_A (Std Dev of Asset A)", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("sigma_B (Std Dev of Asset B)", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("corr_AB (Correlation)", -1.0, 1.0, 0.20, 0.05)

    with col2:
        # Plot frontier in the second (right) column
        plot_two_asset_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
