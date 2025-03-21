import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    Two-Stock Mean–Variance Frontier:
      - If |mu_A - mu_B| < tolerance => Single red point (MVP) as 'Efficient Frontier';
        entire remainder of line is dashed 'inefficient.'
      - Otherwise => standard Markowitz split:
        whichever stock has the higher return determines which side is 'efficient.'
      - Also shows random portfolios in gray for illustration.
    """

    # 1) Parametric frontier for w in [0,1]
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    frontier_returns = []
    frontier_stdevs  = []

    for w in weights:
        # Return:
        r = w * mu_A + (1 - w) * mu_B
        # Variance:
        v = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        frontier_returns.append(r)
        frontier_stdevs.append(np.sqrt(v))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs  = np.array(frontier_stdevs)

    # 2) Find the MVP (lowest std dev)
    idx_min = np.argmin(frontier_stdevs)
    mvp_x = frontier_stdevs[idx_min]
    mvp_y = frontier_returns[idx_min]

    # 3) Random portfolios (optional, for illustration)
    n_random = 3000
    rand_weights = np.random.rand(n_random)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_weights:
        rr = w * mu_A + (1 - w) * mu_B
        rv = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(rr)
        rand_stdevs.append(np.sqrt(rv))

    # 4) Identify user-chosen returns are "equal" or not
    tol = 1e-12
    same_return = (abs(mu_A - mu_B) < tol)

    if same_return:
        # =============== CASE A: SAME RETURNS => ONLY MVP is 'efficient' ===============
        # Everything except MVP => 'inefficient'
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False  # exclude MVP

        inef_x = frontier_stdevs[mask]
        inef_y = frontier_returns[mask]

        # The single 'efficient' point is the MVP
        ef_x = [mvp_x]
        ef_y = [mvp_y]

    else:
        # =============== CASE B: DIFFERENT RETURNS => standard Markowitz split ===============
        # We'll compare the user-chosen returns directly to see which is higher
        # (no need to rely on the param endpoints).
        if mu_A > mu_B:
            # If Stock A has the higher return => from MVP..w=1 is 'efficient'
            inef_x = frontier_stdevs[:idx_min+1]
            inef_y = frontier_returns[:idx_min+1]
            ef_x   = frontier_stdevs[idx_min:]
            ef_y   = frontier_returns[idx_min:]
        else:
            # If Stock B has the higher return => from w=0..MVP is 'efficient'
            ef_x   = frontier_stdevs[:idx_min+1]
            ef_y   = frontier_returns[:idx_min+1]
            inef_x = frontier_stdevs[idx_min:]
            inef_y = frontier_returns[idx_min:]

    # 5) Plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Gray scatter: random portfolios
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # 'Efficient Frontier' => red solid line (or single red point if only 1)
    ax.plot(ef_x, ef_y, 'r-', linewidth=2, label='Efficient Frontier')

    # 'Inefficient' => dashed red
    ax.plot(inef_x, inef_y, 'r--', label='Inefficient Portfolios')

    # Mark MVP with black star
    ax.scatter(mvp_x, mvp_y, marker='*', s=80, color='black', label='Minimum-Variance Portfolio')

    # Mark each stock individually (endpoints in param approach?)
    # But let's rely on user input for naming. We can just put them at:
    #  w=1 => Stock A, w=0 => Stock B from the param arrays:
    std_B = frontier_stdevs[0]   # w=0
    ret_B = frontier_returns[0]
    std_A = frontier_stdevs[-1]  # w=1
    ret_A = frontier_returns[-1]
    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # We want the legend in a specific order:
    handles, labels = ax.get_legend_handles_labels()
    lab2hand = dict(zip(labels, handles))
    desired_order = [
        'Efficient Frontier',
        'Inefficient Portfolios',
        'Random Portfolios',
        'Stock A',
        'Stock B',
        'Minimum-Variance Portfolio'
    ]
    new_handles = [lab2hand[lbl] for lbl in desired_order if lbl in lab2hand]
    new_labels  = [lbl           for lbl in desired_order if lbl in lab2hand]
    ax.legend(new_handles, new_labels, loc='best')

    ax.set_title("Two-Stock Frontier")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier (Single Point If Same Returns)")

    col_sliders, col_chart = st.columns([2, 3])
    with col_sliders:
        st.markdown("### Choose Parameters")
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.09, 0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.09, 0.01)
        sigma_A = st.slider("Std Dev of Stock A", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("Std Dev of Stock B", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("Correlation", -1.0, 1.0, 0.20, 0.05)

    with col_chart:
        plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == '__main__':
    main()
