import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Make the page layout wide so columns can sit side-by-side
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

    # Random simulation (for illustration)
    n_portfolios = 3000
    rand_w = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []

    for w in rand_w:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(p_return)
        rand_stdevs.append(np.sqrt(p_var))

    # Plot with a smaller figure size
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, label='Random Portfolios')
    ax.plot(port_stdevs, port_returns, 'r-', label='Efficient Frontier', linewidth=1)

    # Decrease marker sizes for clarity
    ax.scatter(sigma_A, mu_A, s=40, marker='o', label='Asset A')
    ax.scatter(sigma_B, mu_B, s=40, marker='o', label='Asset B')

    # Shorter labels and smaller font sizes
    ax.set_title('2-Stock Frontier', fontsize=10)
    ax.set_xlabel('Std Dev', fontsize=8)
    ax.set_ylabel('Expect. Return', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.legend(fontsize=7)

    # Force both axes to start at zero
    # ax.set_xlim(left=0)   # x-axis starts at 0
    # ax.set_ylim(bottom=0) # y-axis starts at 0
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Two-Stock Efficient Frontier")

    # Two columns: sliders on the left, chart on the right
    col_sliders, col_chart = st.columns([2, 3])  # Adjust ratio as needed

    with col_sliders:
        st.markdown("### Adjust the Parameters")

        # SIMPLIFIED LABELS
        mu_A = st.slider("Expected Return of Asset A", 0.00, 0.20, 0.10, 0.01)
        mu_B = st.slider("Expected Return of Asset B", 0.00, 0.20, 0.15, 0.01)
        sigma_A = st.slider("Standard Deviation of Asset A", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("Standard Deviation of Asset B", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("Correlation Between Assets A and B", -1.0, 1.0, 0.20, 0.05)

    with col_chart:
        plot_two_asset_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
