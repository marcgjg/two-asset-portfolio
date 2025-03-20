import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_asset_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """Plot 2-asset frontier with bottom portion as a dashed line 
       and top portion as a solid line (the efficient part)."""

    # ----- 1) Compute the required data -----

    # Covariance
    cov_AB = corr_AB * sigma_A * sigma_B

    # Parametric approach: w in [0, 1]
    weights = np.linspace(0, 1, 200)  # more points for a smoother curve
    port_returns = []
    port_stdevs  = []

    for w in weights:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        port_returns.append(p_return)
        port_stdevs.append(np.sqrt(p_var))

    port_returns = np.array(port_returns)
    port_stdevs  = np.array(port_stdevs)

    # Find the index of the minimum variance portfolio
    idx_min = np.argmin(port_stdevs)

    # Split the curve into two segments:
    #   - from w=0 to w at min variance => "inefficient" portion (dashed)
    #   - from w at min variance to w=1 => "efficient" portion (solid)
    x_inef = port_stdevs[:idx_min+1]
    y_inef = port_returns[:idx_min+1]

    x_ef   = port_stdevs[idx_min:]
    y_ef   = port_returns[idx_min:]

    # ----- 2) Random portfolios for illustration -----
    n_portfolios = 3000
    rand_w = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []

    for w in rand_w:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(p_return)
        rand_stdevs.append(np.sqrt(p_var))

    # ----- 3) Plot -----
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, label='Random Portfolios')

    # Plot the "inefficient" (lower) part as dashed
    ax.plot(x_inef, y_inef, 'r--', label='Inefficient')

    # Plot the "efficient" (upper) part as solid
    ax.plot(x_ef, y_ef, 'r-', label='Efficient Frontier', linewidth=2)

    # Mark the individual assets
    ax.scatter(sigma_A, mu_A, s=40, marker='o', label='Asset A')
    ax.scatter(sigma_B, mu_B, s=40, marker='o', label='Asset B')

    ax.set_title('Two-Asset Frontier')
    ax.set_xlabel('Std Dev')
    ax.set_ylabel('Return')
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier with Dashed Inefficient Part")

    col_sliders, col_chart = st.columns([3, 2])

    with col_sliders:
        st.markdown("### Adjust the Parameters")
        mu_A = st.slider("Expected Return of Asset A", 0.00, 0.20, 0.10, 0.01)
        mu_B = st.slider("Expected Return of Asset B", 0.00, 0.20, 0.15, 0.01)
        sigma_A = st.slider("Standard Deviation of Asset A", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("Standard Deviation of Asset B", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("Correlation Between Assets A and B", -1.0, 1.0, 0.20, 0.05)

    with col_chart:
        plot_two_asset_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
