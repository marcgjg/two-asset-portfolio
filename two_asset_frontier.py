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

    # 5) Check if returns match
    tol = 1e-12
    same_return = (abs(muA - muB) < tol)

    if same_return:
        # =========== ALL POINTS except the MVP => 'inefficient' ===========
        mask = np.ones_like(stdevs, dtype=bool)
        mask[idx_min] = False  # exclude MVP

        x_inef = stdevs[mask]
        y_inef = returns[mask]

        # The MVP alone => 'efficient frontier'
        x_ef   = [mvp_x]
        y_ef   = [mvp_y]

        # For clarity, let's color the line black so you see the difference
        # (just to emphasize it's "inefficient"). You can change to red if you like.
        plt.plot(x_inef, y_inef, 'k--', label='Inefficient (same return)')

        # Single point in red for the MVP
        plt.scatter(x_ef, y_ef, c='r', s=80, label='Efficient Frontier (single point)')

    else:
        # =========== Normal logic: whichever stock has higher return => that side is 'efficient' ===========
        if retA > retB:
            # A has higher => from w=0..idx_min is inefficient, from idx_min..1 is efficient
            x_inef = stdevs[:idx_min+1]
            y_inef = returns[:idx_min+1]
            x_ef   = stdevs[idx_min:]
            y_ef   = returns[idx_min:]
        else:
            x_ef   = stdevs[:idx_min+1]
            y_ef   = returns[:idx_min+1]
            x_inef = stdevs[idx_min:]
            y_inef = returns[idx_min:]

        plt.plot(x_ef, y_ef, 'r-', linewidth=2, label='Efficient Frontier')
        plt.plot(x_inef, y_inef, 'r--', label='Inefficient')

    # 6) Plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Gray scatter for random portfolios
    # ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray')

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
