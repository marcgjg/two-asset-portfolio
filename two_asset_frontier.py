import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    If the two returns are the same (within tol):
      - Show ONLY the MVP as a single point (labeled 'Efficient Frontier').
      - Hide the rest of the parametric frontier entirely (no dashed line, nothing).

    Otherwise, standard 2-stock logic:
      - Parametric frontier from w=0..1
      - Split at MVP: one side is 'Efficient Frontier' (solid), the other is 'Inefficient' (dashed).
      - Random portfolios in gray, MVP marked by a black star, etc.
    """

    # 1) Parametric frontier
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    frontier_returns = []
    frontier_stdevs  = []
    for w in weights:
        r = w*mu_A + (1-w)*mu_B
        v = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        frontier_returns.append(r)
        frontier_stdevs.append(np.sqrt(v))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs  = np.array(frontier_stdevs)

    # 2) MVP
    idx_min = np.argmin(frontier_stdevs)
    mvp_x = frontier_stdevs[idx_min]
    mvp_y = frontier_returns[idx_min]

    # 3) Random portfolios
    n_random = 3000
    rand_w = np.random.rand(n_random)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_w:
        rr = w*mu_A + (1 - w)*mu_B
        rv = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(rr)
        rand_stdevs.append(np.sqrt(rv))

    # 4) Check if returns match
    tol = 1e-12
    same_return = abs(mu_A - mu_B) < tol

    # 5) Identify Stock A, Stock B from endpoints
    std_B = frontier_stdevs[0]   # w=0 => B
    ret_B = frontier_returns[0]
    std_A = frontier_stdevs[-1]  # w=1 => A
    ret_A = frontier_returns[-1]

    # 6) Plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Random portfolios in gray
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    if same_return:
        #
        # Case A: EXACT same returns => ONLY show MVP
        #
        #  - We do NOT plot the parametric line at all
        #  - Just a single red dot for 'Efficient Frontier'
        #
        ax.scatter([mvp_x], [mvp_y], color='red', s=70, label='Efficient Frontier')
        # Mark the MVP with a star if you want (overlapping):
        ax.scatter([mvp_x], [mvp_y], marker='*', s=90, color='black', label='Minimum-Variance Portfolio')

    else:
        #
        # Case B: Different returns => standard Markowitz split
        #
        if mu_A > mu_B:
            # from MVP..end => 'efficient'
            x_inef = frontier_stdevs[:idx_min+1]
            y_inef = frontier_returns[:idx_min+1]
            x_ef   = frontier_stdevs[idx_min:]
            y_ef   = frontier_returns[idx_min:]
        else:
            x_ef   = frontier_stdevs[:idx_min+1]
            y_ef   = frontier_returns[:idx_min+1]
            x_inef = frontier_stdevs[idx_min:]
            y_inef = frontier_returns[idx_min:]

        ax.plot(x_ef,   y_ef,   'r-', linewidth=2, label='Efficient Frontier')
        ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')
        # MVP as black star
        ax.scatter([mvp_x], [mvp_y], marker='*', s=90, color='black', label='Minimum-Variance Portfolio')

    # Mark Stock A & Stock B
    ax.scatter(std_A, ret_A, marker='o', s=50, label='Stock A')
    ax.scatter(std_B, ret_B, marker='o', s=50, label='Stock B')

    # Legend order
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
    new_handles = []
    new_labels  = []
    for d in desired_order:
        if d in label2handle:
            new_handles.append(label2handle[d])
            new_labels.append(d)

    ax.legend(new_handles, new_labels, loc='best')

    ax.set_title("Two-Stock Frontier (Hide Param Line if Same Returns)")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier - Single Point if Same Returns")
    col_sliders, col_chart = st.columns([2, 3])

    with col_sliders:
        st.markdown("### Set the Parameters")
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.09, 0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.09, 0.01)
        sigma_A = st.slider("Standard Deviation of Stock A", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("Standard Deviation of Stock B", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("Correlation", -1.0, 1.0, 0.20, 0.05)

    with col_chart:
        plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == '__main__':
    main()
