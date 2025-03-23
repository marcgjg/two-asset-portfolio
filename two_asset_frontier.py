import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    # 1) Parametric frontier
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    frontier_returns = []
    frontier_stdevs  = []

    for w in weights:
        r = w * mu_A + (1 - w) * mu_B
        v = (w**2)*(sigma_A**2) + ((1 - w)**2)*(sigma_B**2) + 2*w*(1 - w)*cov_AB
        frontier_returns.append(r)
        frontier_stdevs.append(np.sqrt(v))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs  = np.array(frontier_stdevs)

    # 2) MVP
    idx_min = np.argmin(frontier_stdevs)
    mvp_x = frontier_stdevs[idx_min]
    mvp_y = frontier_returns[idx_min]

    # 3) Check for same expected returns
    tol = 1e-12
    same_return = abs(mu_A - mu_B) < tol

    fig, ax = plt.subplots(figsize=(8, 4))

    if same_return:
        # =========== Only MVP is efficient ===========
        # Inefficient part = full line minus MVP
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False
        inef_x = frontier_stdevs[mask]
        inef_y = frontier_returns[mask]

        ef_x = [mvp_x]
        ef_y = [mvp_y]

        # Plot
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')
        ax.scatter(ef_x, ef_y, color='red', s=70, label='Efficient Frontier')

    else:
        # =========== Standard logic ===========
        if mu_A > mu_B:
            inef_x = frontier_stdevs[:idx_min+1]
            inef_y = frontier_returns[:idx_min+1]
            ef_x   = frontier_stdevs[idx_min:]
            ef_y   = frontier_returns[idx_min:]
        else:
            ef_x   = frontier_stdevs[:idx_min+1]
            ef_y   = frontier_returns[:idx_min+1]
            inef_x = frontier_stdevs[idx_min:]
            inef_y = frontier_returns[idx_min:]

        ax.plot(ef_x, ef_y, 'r-', linewidth=2, label='Efficient Frontier')
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')

        # Random portfolios
        n_portfolios = 3000
        rand_w = np.random.rand(n_portfolios)
        rand_returns = []
        rand_stdevs  = []
        for w in rand_w:
            r = w * mu_A + (1 - w) * mu_B
            v = (w**2)*(sigma_A**2) + ((1 - w)**2)*(sigma_B**2) + 2*w*(1 - w)*cov_AB
            rand_returns.append(r)
            rand_stdevs.append(np.sqrt(v))

        ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # 4) MVP
    ax.scatter(mvp_x, mvp_y, marker='*', s=90, color='black', label='Minimum-Variance Portfolio')

    # 5) Stock A and B (endpoints)
    std_A = frontier_stdevs[-1]  # w=1
    ret_A = frontier_returns[-1]
    std_B = frontier_stdevs[0]   # w=0
    ret_B = frontier_returns[0]
    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # 6) Legend (ordered and outside)
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
    new_handles = []
    for label in order:
        if label in label2handle:
            new_handles.append(label2handle[label])
    ax.legend(
        new_handles,
        order,
        loc='upper left',
        bbox_to_anchor=(1.04, 1),
        borderaxespad=0,
        prop={'size': 8}
    )

    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    ax.set_title("Two-Stock Frontier (Single Efficient Portfolio if Returns Match)")
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Efficient Frontier")

    col_sliders, col_chart = st.columns([2, 3])

    with col_sliders:
        st.markdown("### Adjust the Parameters")

        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.03, step=0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.03, step=0.01)
        sigma_A = st.slider("Standard Deviation of Stock A", 0.01, 0.40, 0.17, step=0.01)
        sigma_B = st.slider("Standard Deviation of Stock B", 0.01, 0.40, 0.21, step=0.01)
        corr_AB = st.slider("Correlation Coefficient", -1.0, 1.0, 0.60, step=0.05)

    with col_chart:
        plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
