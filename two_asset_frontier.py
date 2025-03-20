import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Use a wide layout so the sliders and chart can appear side-by-side
st.set_page_config(layout='wide')

def plot_two_stock_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    Plots a two-stock mean–variance frontier with:
    - Random portfolios (gray dots).
    - A parametric frontier from w=0 to w=1, split into:
      - 'Efficient Frontier' (solid red) for the higher-return side.
      - 'Inefficient Portfolios' (dashed red) for the lower-return side.
    - Marks Stock A, Stock B, and the minimum-variance portfolio (MVP).
    - Handles the special case where mu_A == mu_B: only the MVP is truly 'efficient'.
    """

    # 1) Basic frontier calculations
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    port_returns = []
    port_stdevs = []

    for w in weights:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        port_returns.append(p_return)
        port_stdevs.append(np.sqrt(p_var))

    port_returns = np.array(port_returns)
    port_stdevs  = np.array(port_stdevs)

    # 2) Identify the minimum-variance portfolio (MVP)
    idx_min = np.argmin(port_stdevs)
    mvp_x = port_stdevs[idx_min]
    mvp_y = port_returns[idx_min]

    # 3) Check if mu_A == mu_B (within a tolerance) => special case
    tol = 1e-12
    same_return = abs(mu_A - mu_B) < tol

    if same_return:
        # Entire parametric line has the same return => only MVP is truly efficient
        x_ef = [mvp_x]
        y_ef = [mvp_y]

        # All other points are 'inefficient'
        x_inef = np.delete(port_stdevs, idx_min)
        y_inef = np.delete(port_returns, idx_min)

    else:
        # Compare returns of Stock B (w=0) and Stock A (w=1)
        ret_B, std_B = port_returns[0],  port_stdevs[0]
        ret_A, std_A = port_returns[-1], port_stdevs[-1]

        # If Stock A has higher return => from MVP to w=1 is 'efficient'
        # else => from w=0 to MVP is 'efficient'
        if ret_A > ret_B:
            x_inef = port_stdevs[:idx_min+1]
            y_inef = port_returns[:idx_min+1]
            x_ef   = port_stdevs[idx_min:]
            y_ef   = port_returns[idx_min:]
        else:
            x_ef   = port_stdevs[:idx_min+1]
            y_ef   = port_returns[:idx_min+1]
            x_inef = port_stdevs[idx_min:]
            y_inef = port_returns[idx_min:]

    # 4) Generate random portfolios for illustration
    n_portfolios = 3000
    rand_weights = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []

    for w in rand_weights:
        rp_return = w * mu_A + (1 - w) * mu_B
        rp_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(rp_return)
        rand_stdevs.append(np.sqrt(rp_var))

    # 5) Plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Random portfolios (gray scatter)
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray')

    # Plot the 'Efficient Frontier' (solid red)
    ax.plot(x_ef, y_ef, 'r-', linewidth=2, label='Efficient Frontier')

    # Plot the 'Inefficient Portfolios' (dashed red)
    ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')

    # Mark each stock individually
    # (Stock A => w=1 => last parametric point,
    #  Stock B => w=0 => first parametric point)
    std_A = port_stdevs[-1]
    ret_A = port_returns[-1]
    std_B = port_stdevs[0]
    ret_B = port_returns[0]
    ax.scatter(std_A, ret_A, s=50, marker='o', label='Stock A')
    ax.scatter(std_B, ret_B, s=50, marker='o', label='Stock B')

    # Mark the MVP with a black star
    ax.scatter(mvp_x, mvp_y, s=80, marker='*', color='black', label='Minimum-Variance Portfolio')

    # Enforce the desired legend order:
    # 1) Efficient Frontier
    # 2) Inefficient Portfolios
    # 3) Stock A
    # 4) Stock B
    # 5) Minimum-Variance Portfolio
    handles, labels = ax.get_legend_handles_labels()

    # Create a mapping label -> handle
    label_to_handle = dict(zip(labels, handles))

    # Reorder them in the desired sequence
    desired_labels = [
        'Efficient Frontier',
        'Inefficient Portfolios',
        'Stock A',
        'Stock B',
        'Minimum-Variance Portfolio'
    ]

    reordered_handles = [label_to_handle[lbl] for lbl in desired_labels]
    ax.legend(reordered_handles, desired_labels, loc='best')

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

        mu_A = st.slider("Expected Return of Stock A", 0.00, 0.20, 0.10, 0.01)
        mu_B = st.slider("Expected Return of Stock B", 0.00, 0.20, 0.15, 0.01)
        sigma_A = st.slider("Standard Deviation of Stock A", 0.01, 0.40, 0.20, 0.01)
        sigma_B = st.slider("Standard Deviation of Stock B", 0.01, 0.40, 0.30, 0.01)
        corr_AB = st.slider("Correlation Between Stocks A and B", -1.0, 1.0, 0.20, 0.05)

    with col_chart:
        plot_two_stock_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
