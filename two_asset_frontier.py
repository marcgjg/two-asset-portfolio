import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    Plots a two-stock frontier with:
      - Random portfolios (gray dots).
      - 'Efficient Frontier' (solid red).
      - 'Inefficient Portfolios' (dashed red).
      - Special case: if Stock A and Stock B have the same return (within a tolerance),
        only the MVP is labeled 'efficient' (a single point).
    """

    # 1) Parametric frontier
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    port_returns = []
    port_stdevs  = []

    for w in weights:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        port_returns.append(p_return)
        port_stdevs.append(np.sqrt(p_var))

    port_returns = np.array(port_returns)
    port_stdevs  = np.array(port_stdevs)

    # 2) Minimum-Variance Portfolio (MVP)
    idx_min = np.argmin(port_stdevs)
    mvp_x   = port_stdevs[idx_min]
    mvp_y   = port_returns[idx_min]

    # 3) Random portfolios (for illustration)
    n_portfolios = 3000
    rand_w = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_w:
        rp_return = w * mu_A + (1 - w) * mu_B
        rp_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(rp_return)
        rand_stdevs.append(np.sqrt(rp_var))

    # 4) Decide how to split efficient vs. inefficient
    # We'll treat the two returns as 'equal' if their absolute difference < some tolerance
    tol = 1e-10
    if abs(mu_A - mu_B) < tol:
        #
        # Same return: entire parametric line is horizontal => ONLY the MVP is truly efficient
        #
        x_ef = [mvp_x]
        y_ef = [mvp_y]

        # The rest are all inefficient
        mask_others = np.ones_like(port_stdevs, dtype=bool)
        mask_others[idx_min] = False  # exclude the MVP
        x_inef = port_stdevs[mask_others]
        y_inef = port_returns[mask_others]

    else:
        #
        # Different returns => normal logic
        #
        # Stock B => w=0
        ret_B, std_B = port_returns[0],  port_stdevs[0]
        # Stock A => w=1
        ret_A, std_A = port_returns[-1], port_stdevs[-1]

        if ret_A > ret_B:
            # A has higher return => from MVP to w=1 is efficient
            x_inef = port_stdevs[:idx_min+1]
            y_inef = port_returns[:idx_min+1]
            x_ef   = port_stdevs[idx_min:]
            y_ef   = port_returns[idx_min:]
        else:
            # B has higher return => from w=0 to MVP is efficient
            x_ef   = port_stdevs[:idx_min+1]
            y_ef   = port_returns[:idx_min+1]
            x_inef = port_stdevs[idx_min:]
            y_inef = port_returns[idx_min:]

    # 5) Stock A, Stock B from param arrays
    # w=0 => B, w=1 => A
    std_B = port_stdevs[0]
    ret_B = port_returns[0]
    std_A = port_stdevs[-1]
    ret_A = port_returns[-1]

    # 6) Plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Random portfolios
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray')

    # 'Efficient Frontier' => solid red
    ax.plot(x_ef, y_ef, 'r-', linewidth=2, label='Efficient Frontier')

    # 'Inefficient Portfolios' => dashed red
    ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')

    # Mark Stock A, Stock B
    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # MVP => star
    ax.scatter(mvp_x, mvp_y, marker='*', s=80, color='black', label='Minimum-Variance Portfolio')

    # Force desired legend order
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    desired_labels = [
        'Efficient Frontier',
        'Inefficient Portfolios',
        'Stock A',
        'Stock B',
        'Minimum-Variance Portfolio'
    ]
    reordered_handles = [label_to_handle[lbl] for lbl in desired_labels]
    ax.legend(reordered_handles, desired_labels, loc='best')

    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Expected Return')
    ax.set_title('Two-Stock Frontier')
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier (Equal Returns => Single Efficient Point)")

    col_sliders, col_chart = st.columns([2, 3])

    with col_sliders:
        st.markdown("### Adjust the Parameters")

        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.10, 0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.10, 0.01)
        sigma_A = st.slider("Standard Deviation of Stock A", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("Standard Deviation of Stock B", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("Correlation Between Stocks A and B", -1.0, 1.0, 0.20, 0.05)

    with col_chart:
        plot_two_stock_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == '__main__':
    main()
