import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    When returns are equal (within tolerance), 
    ONLY the MVP is 'efficient' (a single scatter point).
    The entire rest of the parametric line is dashed as 'Inefficient Portfolios'.
    Otherwise, we do the usual split logic.
    """

    # 1) Generate parametric frontier
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    port_returns = []
    port_stdevs  = []
    for w in weights:
        p_ret = w * mu_A + (1 - w) * mu_B
        p_var = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        port_returns.append(p_ret)
        port_stdevs.append(np.sqrt(p_var))

    port_returns = np.array(port_returns)
    port_stdevs  = np.array(port_stdevs)

    # 2) Random portfolios (for illustration)
    n_portfolios = 3000
    rand_w = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_w:
        r_ret = w * mu_A + (1 - w) * mu_B
        r_var = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(r_ret)
        rand_stdevs.append(np.sqrt(r_var))

    # 3) Minimum-Variance Portfolio
    idx_min = np.argmin(port_stdevs)
    mvp_x   = port_stdevs[idx_min]
    mvp_y   = port_returns[idx_min]

    # 4) Identify Stock A & Stock B
    # w=0 => B, w=1 => A
    std_B = port_stdevs[0]
    ret_B = port_returns[0]
    std_A = port_stdevs[-1]
    ret_A = port_returns[-1]

    # 5) Determine if returns are effectively the same
    tol = 1e-12
    same_return = (abs(mu_A - mu_B) < tol)

    if same_return:
        # ============ CASE: SAME RETURNS => Only MVP is 'Efficient Frontier' ============

        # The entire parametric line (minus MVP) is dashed => 'Inefficient'
        mask = np.ones_like(port_stdevs, dtype=bool)
        mask[idx_min] = False  # exclude MVP
        x_inef = port_stdevs[mask]
        y_inef = port_returns[mask]

        # We'll put just the single MVP point in 'efficient'
        x_ef = [mvp_x]
        y_ef = [mvp_y]

    else:
        # ============ CASE: DIFFERENT RETURNS => Standard logic ============

        # If w=1 => A, w=0 => B
        if ret_A > ret_B:
            # A has higher => from MVP to w=1 is efficient
            x_inef = port_stdevs[:idx_min+1]
            y_inef = port_returns[:idx_min+1]
            x_ef   = port_stdevs[idx_min:]
            y_ef   = port_returns[idx_min:]
        else:
            # B has higher => from w=0 to MVP is efficient
            x_ef   = port_stdevs[:idx_min+1]
            y_ef   = port_returns[:idx_min+1]
            x_inef = port_stdevs[idx_min:]
            y_inef = port_returns[idx_min:]

    # 6) Plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Gray scatter for random portfolios
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray')

    # Solid red => 'Efficient Frontier'
    # NOTE: If same_return == True, x_ef, y_ef is just one point
    ax.plot(x_ef, y_ef, 'r-', linewidth=2, label='Efficient Frontier')

    # Dashed red => 'Inefficient Portfolios'
    ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')

    # Mark Stock A & Stock B
    ax.scatter(std_A, ret_A, s=50, marker='o', label='Stock A')
    ax.scatter(std_B, ret_B, s=50, marker='o', label='Stock B')

    # Mark MVP => black star
    ax.scatter(mvp_x, mvp_y, s=80, marker='*', color='black', label='Minimum-Variance Portfolio')

    # Force legend order
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    desired_labels = [
        'Efficient Frontier',
        'Inefficient Portfolios',
        'Stock A',
        'Stock B',
        'Minimum-Variance Portfolio'
    ]
    new_handles = [label_to_handle[lbl] for lbl in desired_labels]
    ax.legend(new_handles, desired_labels, loc='best')

    ax.set_title("Two-Stock Frontier")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Two-Stock Frontier (Strict Single-Point Efficiency if Returns Match)")

    col_sliders, col_chart = st.columns([2, 3])
    with col_sliders:
        st.markdown("### Adjust the Parameters")
        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.09, 0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.09, 0.01)
        sigma_A = st.slider("Standard Deviation of Stock A", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("Standard Deviation of Stock B", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("Correlation Between Stocks A and B", -1.0, 1.0, 0.20, 0.05)

    with col_chart:
        plot_two_stock_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == '__main__':
    main()
