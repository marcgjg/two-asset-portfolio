import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    Plot the Two-Stock Frontier:
      - If returns are the same, only calculate the MVP (Minimum-Variance Portfolio) and mark it as 'Efficient Frontier'.
      - If returns are different, calculate random portfolios and the efficient/inefficient frontier.
    """

    # 1) Parametric frontier calculation
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

    # 2) Minimum-Variance Portfolio (MVP)
    idx_min = np.argmin(frontier_stdevs)
    mvp_x   = frontier_stdevs[idx_min]
    mvp_y   = frontier_returns[idx_min]

    # 3) If returns are the same, only calculate MVP and plot it as the single efficient portfolio
    tol = 1e-12
    same_return = (abs(mu_A - mu_B) < tol)

    fig, ax = plt.subplots(figsize=(8, 4))

    if same_return:
        # Only the MVP is "efficient"
        ax.scatter([mvp_x], [mvp_y], color='red', s=70, label='Efficient Frontier')

        # The rest of the frontier is inefficient, plotted as dashed line
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False  # exclude MVP
        x_inef = frontier_stdevs[mask]
        y_inef = frontier_returns[mask]

        ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')

    else:
        # Calculate random portfolios (for different returns only)
        n_portfolios = 3000
        rand_w = np.random.rand(n_portfolios)
        rand_returns = []
        rand_stdevs  = []
        for w in rand_w:
            rp_ret = w * mu_A + (1 - w) * mu_B
            rp_var = (w**2)*(sigma_A**2) + ((1 - w)**2)*(sigma_B**2) + 2*w*(1 - w)*cov_AB
            rand_returns.append(rp_ret)
            rand_stdevs.append(np.sqrt(rp_var))

        # Efficient/inefficient split logic for different returns
        if mu_A > mu_B:
            # From MVP..end is efficient, from start..MVP is inefficient
            inef_x = frontier_stdevs[:idx_min+1]
            inef_y = frontier_returns[:idx_min+1]
            ef_x   = frontier_stdevs[idx_min:]
            ef_y   = frontier_returns[idx_min:]
        else:
            # From start..MVP is efficient, from MVP..end is inefficient
            ef_x   = frontier_stdevs[:idx_min+1]
            ef_y   = frontier_returns[:idx_min+1]
            inef_x = frontier_stdevs[idx_min:]
            inef_y = frontier_returns[idx_min:]

        ax.plot(ef_x, ef_y, 'r-', linewidth=2, label='Efficient Frontier')
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')

        # Plot random portfolios in gray
        ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # Mark Stock A & Stock B (w=1 for A, w=0 for B)
    std_B = frontier_stdevs[0]
    ret_B = frontier_returns[0]
    std_A = frontier_stdevs[-1]
    ret_A = frontier_returns[-1]
    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # Mark MVP (Minimum-Variance Portfolio) with a star
    ax.scatter(mvp_x, mvp_y, marker='*', s=90, color='black', label='Minimum-Variance Portfolio')

    # 4) Force legend order and place the legend outside
    handles, labels = ax.get_legend_handles_labels()
    label2handle = dict(zip(labels, handles))
    desired_order = [
        'Efficient Frontier',
        'Inefficient Portfolios',
        'Random Portfolios',
        'Stock A',
        'Stock B',
        'Minimum-Variance Portfolio'
    ]
    new_handles, new_labels = [], []
    for lbl in desired_order:
        if lbl in label2handle:
            new_handles.append(label2handle[lbl])
            new_labels.append(lbl)

    ax.legend(
        new_handles,
        new_labels,
        loc='upper left',
        bbox_to_anchor=(1.04, 1),
        borderaxespad=0,
        prop={'size': 8}  # smaller font for the legend
    )

    ax.set_title("Two-Stock Frontier")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier")

    col_sliders, col_chart = st.columns([2, 3])

    with col_sliders:
        st.markdown("### Adjust the Parameters")
        # Sliders with step=0.01 for discrete increments
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.09, step=0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.09, step=0.01)
        sigma_A = st.slider("Standard Deviation of Stock A", 0.01, 0.40, 0.12, step=0.01)
        sigma_B = st.slider("Standard Deviation of Stock B", 0.01, 0.40, 0.15, step=0.01)
        corr_AB = st.slider("Correlation Between Stocks A and B", -1.0, 1.0, -0.70, step=0.05)

    with col_chart:
        plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
