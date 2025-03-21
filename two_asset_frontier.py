import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    Two-Stock Frontier:
      - If |mu_A - mu_B| < tol => both have 'same return':
         * Show the entire horizontal line as dashed (inefficient).
         * EXCEPT the MVP, which is a single red dot labeled 'Efficient Frontier'.
      - Otherwise => normal Markowitz logic:
         * A dashed portion for inefficient,
         * A solid portion for efficient.
      - Also random portfolios in gray, plus Stock A, Stock B, and MVP star.
    """

    # 1) Parametric frontier for w in [0,1]
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

    # 3) Random portfolios (for illustration)
    n_portfolios = 3000
    rand_weights = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_weights:
        rp_ret = w * mu_A + (1 - w) * mu_B
        rp_var = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(rp_ret)
        rand_stdevs.append(np.sqrt(rp_var))

    # 4) Identify Stock A & Stock B from the param arrays
    #    (w=0 => B, w=1 => A)
    std_B = frontier_stdevs[0]
    ret_B = frontier_returns[0]
    std_A = frontier_stdevs[-1]
    ret_A = frontier_returns[-1]

    # 5) Check if returns are effectively equal
    tol = 1e-12
    same_return = abs(mu_A - mu_B) < tol

    # We'll store the 'efficient' x,y in ef_x, ef_y
    # and 'inefficient' in inef_x, inef_y
    # Then we'll plot accordingly
    if same_return:
        #
        # ========== CASE: Both returns match => entire line is same return ========== 
        # We'll label the entire param line as 'inefficient' EXCEPT the MVP
        #
        # 1) Create a mask that excludes the MVP from that line
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False  # exclude MVP

        inef_x = frontier_stdevs[mask]
        inef_y = frontier_returns[mask]

        # 2) MVP alone => 'Efficient Frontier'
        ef_x = [mvp_x]
        ef_y = [mvp_y]

    else:
        #
        # ========== CASE: Different returns => normal Markowitz logic ==========
        #
        if mu_A > mu_B:
            # If Stock A has higher return => from MVP..end is efficient
            inef_x = frontier_stdevs[:idx_min+1]
            inef_y = frontier_returns[:idx_min+1]
            ef_x   = frontier_stdevs[idx_min:]
            ef_y   = frontier_returns[idx_min:]
        else:
            ef_x   = frontier_stdevs[:idx_min+1]
            ef_y   = frontier_returns[:idx_min+1]
            inef_x = frontier_stdevs[idx_min:]
            inef_y = frontier_returns[idx_min:]

    # 6) Plot 
    fig, ax = plt.subplots(figsize=(5, 3))

    # Gray scatter => random portfolios
    # ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    if same_return:
        # Entire line except MVP => 'inefficient' dashed
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')

        # MVP => single red dot for 'Efficient Frontier'
        # (no line, just a dot)
        ax.scatter(ef_x, ef_y, color='red', s=60, label='Efficient Frontier')

        # Also mark MVP with black star (overlap same point)
        ax.scatter(ef_x, ef_y, marker='*', s=80, color='black', label='Minimum-Variance Portfolio')

    else:
        # normal logic => solid portion for 'efficient', dashed for 'inefficient'
        ax.plot(ef_x, ef_y, 'r-', linewidth=2, label='Efficient Frontier')
        ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')

        # Mark MVP with black star
        ax.scatter([mvp_x], [mvp_y], s=80, marker='*', color='black', label='Minimum-Variance Portfolio')

    # Stock A, Stock B
    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # Force Legend Order
    handles, labels = ax.get_legend_handles_labels()
    label2handle = dict(zip(labels, handles))

    # Just in case some labels aren't present in a scenario
    desired = [
        'Efficient Frontier',
        'Inefficient Portfolios',
        'Random Portfolios',
        'Stock A',
        'Stock B',
        'Minimum-Variance Portfolio'
    ]
    new_handles = []
    new_labels  = []
    for d in desired:
        if d in label2handle:
            new_handles.append(label2handle[d])
            new_labels.append(d)

    ax.legend(new_handles, new_labels, loc='best')

    ax.set_title("Two-Stock Frontier")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    plt.tight_layout()

    # Example: place legend to the right
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0)

    # Possibly increase figure size
    # fig.set_size_inches(7, 4)

    plt.tight_layout()
    st.pyplot(fig)

    
    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier")

    col_sliders, col_chart = st.columns([2, 3])
    with col_sliders:
        st.markdown("### Adjust the Parameters")
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.09, 0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.09, 0.01)
        sigma_A = st.slider("Standard Deviation of Stock A", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("Standard Deviation of Stock B", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("Correlation", -1.0, 1.0, 0.20, 0.05)

    with col_chart:
        plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == '__main__':
    main()
