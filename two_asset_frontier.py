import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    cov_AB = corr_AB * sigma_A * sigma_B
    weights = np.linspace(0, 1, 200)
    frontier_returns = []
    frontier_stdevs = []

    for w in weights:
        r = w * mu_A + (1 - w) * mu_B
        v = (w**2)*(sigma_A**2) + ((1 - w)**2)*(sigma_B**2) + 2*w*(1 - w)*cov_AB
        frontier_returns.append(r)
        frontier_stdevs.append(np.sqrt(v))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs = np.array(frontier_stdevs)

    # Minimum-variance portfolio (MVP)
    idx_min = np.argmin(frontier_stdevs)
    mvp_x = frontier_stdevs[idx_min]
    mvp_y = frontier_returns[idx_min]

    # Round to avoid float mismatch
    rounded_mu_A = round(mu_A, 6)
    rounded_mu_B = round(mu_B, 6)
    tol = 1e-12
    same_return = abs(rounded_mu_A - rounded_mu_B) < tol

    fig, ax = plt.subplots(figsize=(8, 4))

    if same_return:
        # Inefficient: entire line except MVP
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False
        inef_x = frontier_stdevs[mask]
        inef_y = frontier_returns[mask]

        # Efficient: just the MVP (single red dot!)
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')
        ax.scatter([mvp_x], [mvp_y], color='red', s=70, label='Efficient Frontier')

    else:
        # Markowitz logic
        if mu_A > mu_B:
            inef_x = frontier_stdevs[:idx_min+1]
            inef_y = frontier_returns[:idx_min+1]
            ef_x = frontier_stdevs[idx_min:]
            ef_y = frontier_returns[idx_min:]
        else:
            ef_x = frontier_stdevs[:idx_min+1]
            ef_y = frontier_returns[:idx_min+1]
            inef_x = frontier_stdevs[idx_min:]
            inef_y = frontier_returns[idx_min:]

        ax.plot(ef_x, ef_y, 'r-', linewidth=2, label='Efficient Frontier')
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')

        # Random portfolios
        rand_w = np.random.rand(3000)
        rand_returns = []
        rand_stdevs = []
        for w in rand_w:
            r = w * mu_A + (1 - w) * mu_B
            v = (w**2)*(sigma_A**2) + ((1 - w)**2)*(sigma_B**2) + 2*w*(1 - w)*cov_AB
            rand_returns.append(r)
            rand_stdevs.append(np.sqrt(v))
        ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # MVP star
    ax.scatter([mvp_x], [mvp_y], marker='*', s=90, color='black', label='Minimum-Variance Portfolio')

    # Stocks
    ax.scatter(frontier_stdevs[-1], frontier_returns[-1], marker='o', s=50, label='Stock A')
    ax.scatter(frontier_stdevs[0], frontier_returns[0], marker='o', s=50, label='Stock B')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    label2handle = dict(zip(labels, handles))
    order = [
        'Efficient Frontier',
        'Inefficient Portfolios',
        'Random Portfolios',
        'Stock A',
        'Stock B',
        'Minimum-Variance Portfolio'
    ]
    new_handles = [label2handle[lbl] for lbl in order if lbl in label2handle]
    ax.legend(new_handles, order, loc='upper left', bbox_to_anchor=(1.04, 1), prop={'size': 8})

    ax.set_title("Two-Stock Frontier (Only MVP Efficient if Returns Match)")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Two-Stock Efficient Frontier")

    col_sliders, col_chart = st.columns([2, 3])
    with col_sliders:
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.03, step=0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.03, step=0.01)
        sigma_A = st.slider("Standard Deviation of Stock A", 0.01, 0.40, 0.15, step=0.01)
        sigma_B = st.slider("Standard Deviation of Stock B", 0.01, 0.40, 0.25, step=0.01)
        corr_AB = st.slider("Correlation Coefficient", -1.0, 1.0, -0.40, step=0.05)

        rounded_mu_A = round(mu_A, 6)
        rounded_mu_B = round(mu_B, 6)
        same_return = abs(rounded_mu_A - rounded_mu_B) < 1e-12

        st.write(f"mu_A = {mu_A}, mu_B = {mu_B}, difference = {mu_A - mu_B}")
        if same_return:
            st.success("✅ Returns are considered identical. Only MVP is efficient.")
        else:
            st.info("ℹ️ Returns are different. Standard efficient frontier will be shown.")

    with col_chart:
        plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
