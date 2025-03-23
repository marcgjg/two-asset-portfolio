import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_frontier(mu_A, mu_B, sigma_A, sigma_B, corr):
    cov_AB = corr * sigma_A * sigma_B
    w = np.linspace(0, 1, 200)

    returns = w * mu_A + (1 - w) * mu_B
    variances = (
        (w**2) * sigma_A**2 +
        (1 - w)**2 * sigma_B**2 +
        2 * w * (1 - w) * cov_AB
    )
    stdevs = np.sqrt(variances)

    idx_mvp = np.argmin(stdevs)
    mvp_x = float(stdevs[idx_mvp])
    mvp_y = float(returns[idx_mvp])

    same_return = abs(round(mu_A, 10) - round(mu_B, 10)) < 1e-10

    fig, ax = plt.subplots(figsize=(8, 4))

    if same_return:
        # Show dashed line for all but MVP
        mask = np.arange(len(stdevs)) != idx_mvp
        ax.plot(stdevs[mask], returns[mask], 'r--', label='Inefficient Portfolios')
        ax.scatter([mvp_x], [mvp_y], color='red', s=70, label='Efficient Frontier')
    else:
        # Efficient and inefficient segments
        if mu_A > mu_B:
            ax.plot(stdevs[:idx_mvp+1], returns[:idx_mvp+1], 'r--', label='Inefficient Portfolios')
            ax.plot(stdevs[idx_mvp:], returns[idx_mvp:], 'r-', label='Efficient Frontier')
        else:
            ax.plot(stdevs[:idx_mvp+1], returns[:idx_mvp+1], 'r-', label='Efficient Frontier')
            ax.plot(stdevs[idx_mvp:], returns[idx_mvp:], 'r--', label='Inefficient Portfolios')

        # Random portfolios
        rand_w = np.random.rand(3000)
        rand_r = rand_w * mu_A + (1 - rand_w) * mu_B
        rand_v = (
            rand_w**2 * sigma_A**2 +
            (1 - rand_w)**2 * sigma_B**2 +
            2 * rand_w * (1 - rand_w) * cov_AB
        )
        rand_s = np.sqrt(rand_v)
        ax.scatter(rand_s, rand_r, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # MVP as star
    ax.scatter([mvp_x], [mvp_y], marker='*', s=100, color='black', label='Minimum-Variance Portfolio')

    # Endpoints: Stock A (w=1), Stock B (w=0)
    ax.scatter([stdevs[0]], [returns[0]], marker='o', s=50, label='Stock B (100%)')
    ax.scatter([stdevs[-1]], [returns[-1]], marker='o', s=50, label='Stock A (100%)')

    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    ax.set_title("Two-Stock Frontier (Single Efficient Point if Returns Match)")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.04, 1), prop={'size': 8})
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Two-Stock Efficient Frontier")

    col_sliders, col_plot = st.columns([2, 3])
    with col_sliders:
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.03, step=0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.03, step=0.01)
        sigma_A = st.slider("Std Dev of Stock A", 0.01, 0.40, 0.15, step=0.01)
        sigma_B = st.slider("Std Dev of Stock B", 0.01, 0.40, 0.25, step=0.01)
        corr_AB = st.slider("Correlation Coefficient", -1.0, 1.0, -0.40, step=0.05)

        st.write(f"mu_A = {mu_A}, mu_B = {mu_B}, Δ = {mu_A - mu_B}")

    with col_plot:
        plot_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
