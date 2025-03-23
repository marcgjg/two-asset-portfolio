import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def compute_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    cov_AB = corr_AB * sigma_A * sigma_B
    weights = np.linspace(0, 1, 200)
    rets, stds = [], []

    for w in weights:
        r = w * mu_A + (1 - w) * mu_B
        v = (
            (w**2) * sigma_A**2 +
            ((1 - w)**2) * sigma_B**2 +
            2 * w * (1 - w) * cov_AB
        )
        rets.append(r)
        stds.append(np.sqrt(v))

    rets = np.array(rets)
    stds = np.array(stds)
    return weights, rets, stds

def plot_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    weights, returns, stds = compute_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)
    idx_mvp = np.argmin(stds)
    mvp_x = stds[idx_mvp]
    mvp_y = returns[idx_mvp]

    tol = 1e-12
    same_return = abs(round(mu_A, 6) - round(mu_B, 6)) < tol

    fig, ax = plt.subplots(figsize=(8, 4))

    if same_return:
        st.write("🔴 Same returns — single efficient portfolio (MVP).")
        # Plot inefficient frontier (all points except MVP)
        mask = np.arange(len(stds)) != idx_mvp
        ax.plot(stds[mask], returns[mask], 'r--', label='Inefficient Portfolios')
        # Plot MVP as red dot
        ax.scatter(mvp_x, mvp_y, color='red', s=80, label='Efficient Frontier')
    else:
        st.write("🟢 Different returns — full efficient frontier.")
        if mu_A > mu_B:
            x_ef = stds[idx_mvp:]
            y_ef = returns[idx_mvp:]
            x_inef = stds[:idx_mvp+1]
            y_inef = returns[:idx_mvp+1]
        else:
            x_ef = stds[:idx_mvp+1]
            y_ef = returns[:idx_mvp+1]
            x_inef = stds[idx_mvp:]
            y_inef = returns[idx_mvp:]

        ax.plot(x_ef, y_ef, 'r-', linewidth=2, label='Efficient Frontier')
        ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')

        # Add random portfolios
        n = 3000
        rand_w = np.random.rand(n)
        rand_returns = rand_w * mu_A + (1 - rand_w) * mu_B
        rand_vars = (
            rand_w**2 * sigma_A**2 +
            (1 - rand_w)**2 * sigma_B**2 +
            2 * rand_w * (1 - rand_w) * corr_AB * sigma_A * sigma_B
        )
        rand_stds = np.sqrt(rand_vars)
        ax.scatter(rand_stds, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # MVP star
    ax.scatter(mvp_x, mvp_y, marker='*', s=100, color='black', label='Minimum-Variance Portfolio')

    # Stocks
    ax.scatter(stds[0], returns[0], marker='o', s=50, label='Stock B (100%)')
    ax.scatter(stds[-1], returns[-1], marker='o', s=50, label='Stock A (100%)')

    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier: Single Dot if Returns Match")
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), prop={'size': 8})
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
        corr_AB = st.slider("Correlation", -1.0, 1.0, -0.40, step=0.05)

        st.write(f"mu_A = {mu_A}, mu_B = {mu_B}, Δ = {mu_A - mu_B}")

    with col_plot:
        plot_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
