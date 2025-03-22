import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    Plots a standard two-stock Markowitz frontier:
      - random portfolios (gray scatter)
      - parametric frontier from w=0..1
      - identifies a 'minimum variance portfolio' (MVP)
      - splits the line into 'efficient frontier' (solid) vs. 'inefficient' (dashed)
        based on which stock has the higher return.
      - the legend is placed outside the plot so it doesn't obscure the data.
    """

    # 1) Parametric Frontier
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    frontier_returns = []
    frontier_stdevs  = []

    for w in weights:
        p_ret = w * mu_A + (1 - w) * mu_B
        p_var = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        frontier_returns.append(p_ret)
        frontier_stdevs.append(np.sqrt(p_var))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs  = np.array(frontier_stdevs)

    # 2) Minimum-Variance Portfolio (MVP)
    idx_min = np.argmin(frontier_stdevs)
    mvp_x   = frontier_stdevs[idx_min]
    mvp_y   = frontier_returns[idx_min]

    # 3) Random Portfolios
    n_portfolios = 3000
    rand_w = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_w:
        rp_return = w * mu_A + (1 - w) * mu_B
        rp_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(rp_return)
        rand_stdevs.append(np.sqrt(rp_var))

    # 4) Figure out which side is 'efficient' vs. 'inefficient'
    #    by comparing mu_A vs. mu_B
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

    # 5) Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Random portfolios (gray scatter)
    # ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # Efficient vs. Inefficient lines
    ax.plot(ef_x, ef_y, 'r-', linewidth=2, label='Efficient Frontier')
    ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')

    # MVP = black star
    ax.scatter([mvp_x], [mvp_y], marker='*', s=90, color='black', label='Minimum-Variance Portfolio')

    # Stock A (w=1) & Stock B (w=0)
    std_B = frontier_stdevs[0]    # w=0
    ret_B = frontier_returns[0]
    std_A = frontier_stdevs[-1]   # w=1
    ret_A = frontier_returns[-1]
    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # 6) Force Legend Order
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
        prop={'size': 8}  # smaller font
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
        # Sliders with step=0.01
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.09, step=0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.10, step=0.01)
        sigma_A = st.slider("Std Dev of Stock A", 0.01, 0.40, 0.20, step=0.01)
        sigma_B = st.slider("Std Dev of Stock B", 0.01, 0.40, 0.30, step=0.01)
        corr_AB = st.slider("Correlation", -1.0, 1.0, 0.20, step=0.05)

        # 1) We only want to forbid EXACT same returns (like 0.09 vs. 0.09).
        #    If you prefer a small tolerance, you can check abs(mu_A - mu_B) < ...
        if mu_A == mu_B:
            st.error("You cannot pick the same expected return for both stocks. Please adjust one of them.")
            st.stop()

    # 2) If code gets here, returns are definitely different
    with col_chart:
        plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
