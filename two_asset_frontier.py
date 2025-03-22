import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    If mu_A == mu_B (within tolerance):
       * entire line is labeled 'Inefficient' (dashed)
       * the MVP (lowest std dev) is a single red dot for 'Efficient Frontier'
    Otherwise:
       * standard 2-stock logic (one portion is solid 'efficient', the other dashed 'inefficient').
    Legend placed outside, wide figure, smaller font, random portfolios in gray for illustration.
    """

    # 1) Parametric Frontier: w in [0,1]
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    frontier_returns = []
    frontier_stdevs  = []

    for w in weights:
        r = w * mu_A + (1 - w) * mu_B
        v = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        frontier_returns.append(r)
        frontier_stdevs.append(np.sqrt(v))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs  = np.array(frontier_stdevs)

    # 2) Minimum-Variance Portfolio (MVP)
    idx_min = np.argmin(frontier_stdevs)
    mvp_x   = frontier_stdevs[idx_min]
    mvp_y   = frontier_returns[idx_min]

    # 3) Random portfolios
    n_portfolios = 3000
    rand_w = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_w:
        rp_return = w * mu_A + (1 - w) * mu_B
        rp_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(rp_return)
        rand_stdevs.append(np.sqrt(rp_var))

    # 4) Check if returns are effectively the same
    tol = 1e-12
    same_return = (abs(mu_A - mu_B) < tol)

    if same_return:
        # ========== CASE: Stocks have identical returns => only MVP is truly efficient ==========

        # The entire parametric line => 'inefficient' except MVP
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False
        inef_x = frontier_stdevs[mask]
        inef_y = frontier_returns[mask]

        # The MVP => single red dot labeled 'Efficient Frontier'
        ef_x = [mvp_x]
        ef_y = [mvp_y]

    else:
        # ========== CASE: Different returns => normal 2-stock logic ==========

        # We'll see which stock has the higher return from user input
        if mu_A > mu_B:
            # from w=0..MVP => 'inefficient', from MVP..w=1 => 'efficient'
            inef_x = frontier_stdevs[:idx_min+1]
            inef_y = frontier_returns[:idx_min+1]
            ef_x   = frontier_stdevs[idx_min:]
            ef_y   = frontier_returns[idx_min:]
        else:
            ef_x   = frontier_stdevs[:idx_min+1]
            ef_y   = frontier_returns[:idx_min+1]
            inef_x = frontier_stdevs[idx_min:]
            inef_y = frontier_returns[idx_min:]

    # 5) Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Random portfolios (gray)
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    if same_return:
        # entire line => dashed 'inefficient'
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')
        # single red dot => 'Efficient Frontier'
        ax.scatter(ef_x, ef_y, color='red', s=70, label='Efficient Frontier')
        # MVP also has black star if you want
        ax.scatter(ef_x, ef_y, marker='*', s=90, color='black', label='Minimum-Variance Portfolio')
    else:
        # normal logic => solid portion for 'efficient', dashed for 'inefficient'
        ax.plot(ef_x, ef_y, 'r-', linewidth=2, label='Efficient Frontier')
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')
        ax.scatter([mvp_x], [mvp_y], marker='*', s=90, color='black', label='Minimum-Variance Portfolio')

    # Mark Stock A & Stock B from param endpoints (w=1 => A, w=0 => B)
    std_B = frontier_stdevs[0]
    ret_B = frontier_returns[0]
    std_A = frontier_stdevs[-1]
    ret_A = frontier_returns[-1]

    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # 6) Legend order
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
        prop={'size': 8}  # smaller legend font
    )

    ax.set_title("Two-Stock Frontier (Single MVP if Same Returns)")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier: Single Efficient Portfolio if Returns Match")

    col_sliders, col_chart = st.columns([2, 3])
    with col_sliders:
        st.markdown("### Adjust the Parameters (0.01 step for returns)")

        # Sliders with step=0.01, allowing user to pick same or different returns
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.09, step=0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.09, step=0.01)
        sigma_A = st.slider("Std Dev of Stock A", 0.01, 0.40, 0.20, step=0.01)
        sigma_B = st.slider("Std Dev of Stock B", 0.01, 0.40, 0.30, step=0.01)
        corr_AB = st.slider("Correlation", -1.0, 1.0, 0.20, step=0.05)

    with col_chart:
        plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
