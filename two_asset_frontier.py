import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    cov_AB = corr_AB * sigma_A * sigma_B
    weights = np.linspace(0, 1, 200)
    returns, stdevs = [], []

    for w in weights:
        r = w * mu_A + (1 - w) * mu_B
        v = (w**2)*sigma_A**2 + ((1 - w)**2)*sigma_B**2 + 2*w*(1 - w)*cov_AB
        returns.append(r)
        stdevs.append(np.sqrt(v))

    returns = np.array(returns)
    stdevs = np.array(stdevs)

    idx_mvp = np.argmin(stdevs)
    mvp_x = stdevs[idx_mvp]
    mvp_y = returns[idx_mvp]

    # Compare rounded values
    tol = 1e-12
    same_return = abs(round(mu_A, 6) - round(mu_B, 6)) < tol

    fig, ax = plt.subplots(figsize=(8, 4))

    if same_return:
        # DEBUG: Show we’re using single-point logic
        st.write("🔴 Same return detected — plotting a single red dot.")
        # Plot dashed line (inefficient)
        mask = np.ones_like(stdevs, dtype=bool)
        mask[idx_mvp] = False
        ax.plot(stdevs[mask], returns[mask], 'r--', label='Inefficient Portfolios')
        # Plot MVP as red dot
        ax.scatter([mvp_x], [mvp_y], color='red', s=70, label='Efficient Frontier')
    else:
        st.write("🟢 Returns differ — drawing solid and dashed segments.")
        if mu_A > mu_B:
            ax.plot(stdevs[:idx_mvp+1], returns[:idx_mvp+1], 'r--', label='Inefficient Portfolios')
            ax.plot(stdevs[idx_mvp:], returns[idx_mvp:], 'r-', label='Efficient Frontier')
        else:
            ax.plot(stdevs[:idx_mvp+1], returns[:idx_mvp+1], 'r-', label='Efficient Frontier')
            ax.plot(stdevs[idx_mvp:], returns[idx_mvp:], 'r--', label='Inefficient Portfolios')
        # Add random portfolios
        rand_stdevs, rand_returns = [], []
        for _ in range(3000):
            w = np.random.rand()
            r = w * mu_A + (1 - w) * mu_B
            v = (w**2)*sigma_A**2 + ((1 - w)**2)*sigma_B**2 + 2*w*(1 - w)*cov_AB
            rand_returns.append(r)
            rand_stdevs.append(np.sqrt(v))
        ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # MVP marker
    ax.scatter([mvp_x], [mvp_y], marker='*', s=100, color='black', label='Minimum-Variance Portfolio')
    # Stocks A and B
    ax.scatter(stdevs[0], returns[0], marker='o', s=50, label='Stock B (100%)')
    ax.scatter(stdevs[-1], returns[-1], marker='o', s=50, label='Stock A (100%)')

    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier — Single Dot if Returns Match")
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), prop={'size': 8})
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("🎯 Two-Stock Efficient Frontier")

    mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.03, step=0.01)
    mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.03, step=0.01)
    sigma_A = st.slider("Std Dev of Stock A", 0.01, 0.40, 0.15, step=0.01)
    sigma_B = st.slider("Std Dev of Stock B", 0.01, 0.40, 0.25, step=0.01)
    corr_AB = st.slider("Correlation Coefficient", -1.0, 1.0, -0.40, step=0.05)

    st.write(f"mu_A = {mu_A}, mu_B = {mu_B}, difference = {mu_A - mu_B}")
    plot_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
