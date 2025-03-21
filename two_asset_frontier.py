import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    If |mu_A - mu_B| < tol => same returns => 
      - plot only ONE red dot (the MVP) for the 'Efficient Frontier'
      - the rest of the parametric line is dashed red ('Inefficient')
    Otherwise => standard two-stock logic:
      - dashed portion is 'inefficient', 
      - solid portion is 'efficient' 
      - depending on which stock has the higher return.
    Also shows random portfolios in gray for illustration.
    """

    # 1) Parametric frontier
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    frontier_returns = []
    frontier_stdevs  = []
    for w in weights:
        r = w*mu_A + (1-w)*mu_B
        v = (w**2)*sigma_A**2 + ((1-w)**2)*sigma_B**2 + 2*w*(1-w)*cov_AB
        frontier_returns.append(r)
        frontier_stdevs.append(np.sqrt(v))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs  = np.array(frontier_stdevs)

    # 2) Find Minimum-Variance Portfolio (MVP)
    idx_min = np.argmin(frontier_stdevs)
    mvp_x = frontier_stdevs[idx_min]
    mvp_y = frontier_returns[idx_min]

    # 3) Random portfolios
    n_random = 3000
    rand_w = np.random.rand(n_random)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_w:
        rr = w * mu_A + (1 - w) * mu_B
        rv = (w**2)*sigma_A**2 + ((1-w)**2)*sigma_B**2 + 2*w*(1-w)*cov_AB
        rand_returns.append(rr)
        rand_stdevs.append(np.sqrt(rv))

    # 4) Distinguish same-return vs. different-return
    tol = 1e-12
    same_return = abs(mu_A - mu_B) < tol

    if same_return:
        # ======= CASE A: SAME RETURNS => ONLY MVP is 'efficient' =======
        # The rest are labeled 'inefficient'
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False  # exclude MVP from the line
        x_inef = frontier_stdevs[mask]
        y_inef = frontier_returns[mask]

        # We'll do a SCATTER for the single efficient point
        x_ef = [mvp_x]
        y_ef = [mvp_y]

        # We'll plot the entire line (minus MVP) as dashed red = 'inefficient'
        # The single point as a red dot for 'efficient'
        line_mode = 'dashed_single_point'
    else:
        # ======= CASE B: DIFFERENT RETURNS => normal Markowitz split =======
        line_mode = 'normal'
        # Compare user-chosen returns directly
        if mu_A > mu_B:
            # from MVP..end is 'efficient'
            x_inef = frontier_stdevs[:idx_min+1]
            y_inef = frontier_returns[:idx_min+1]
            x_ef   = frontier_stdevs[idx_min:]
            y_ef   = frontier_returns[idx_min:]
        else:
            # from start..MVP is 'efficient'
            x_ef   = frontier_stdevs[:idx_min+1]
            y_ef   = frontier_returns[:idx_min+1]
            x_inef = frontier_stdevs[idx_min:]
            y_inef = frontier_returns[idx_min:]

    # 5) Identify the user-labeled stocks (endpoints)
    std_B = frontier_stdevs[0]   # w=0 => B
    ret_B = frontier_returns[0]
    std_A = frontier_stdevs[-1]  # w=1 => A
    ret_A = frontier_returns[-1]

    # 6) Plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Gray scatter for random
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    if line_mode == 'dashed_single_point':
        # Plot all param combos (minus MVP) as a dashed line
        ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')
        # Just one red dot for the MVP => 'Efficient Frontier'
        ax.scatter([mvp_x], [mvp_y], color='red', s=70, label='Efficient Frontier')
    else:
        # normal case => we have x_ef,y_ef for the efficient line, x_inef,y_inef for inefficient
        ax.plot(x_ef,   y_ef,   'r-', linewidth=2, label='Efficient Frontier')
        ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')

    # Mark the MVP with a star (in black)
    ax.scatter(mvp_x, mvp_y, s=90, marker='*', color='black', label='Minimum-Variance Portfolio')

    # Mark Stock A, Stock B
    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # Enforce a particular legend order
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    desired = [
        'Efficient Frontier',
        'Inefficient Portfolios',
        'Random Portfolios',
        'Stock A',
        'Stock B',
        'Minimum-Variance Portfolio'
    ]
    # Some labels may not exist if same_return => 'Efficient Frontier' is a scatter
    new_handles = []
    new_labels = []
    for d in desired:
        if d in label_to_handle:
            new_handles.append(label_to_handle[d])
            new_labels.append(d)

    ax.legend(new_handles, new_labels, loc='best')

    ax.set_title("Two-Stock Frontier")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier (Single Scatter If Same Returns)")

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
