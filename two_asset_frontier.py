import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_efficient_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    Plots the two-stock frontier in mean–variance space,
    labeling the portion below the MVP (in return) as 'Inefficient' (dashed)
    and the portion above the MVP as 'Efficient Frontier' (solid).
    Also marks the Minimum-Variance Portfolio with a star.
    """

    # ---------- 1) Compute the parametric frontier ----------

    # Covariance between the two stocks
    cov_AB = corr_AB * sigma_A * sigma_B

    # We let w vary from 0 to 1 (no short selling).
    # w=1 => 100% in Stock A, w=0 => 100% in Stock B
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

    # ---------- 2) Find the Minimum-Variance Portfolio (lowest stdev) ----------
    idx_min = np.argmin(port_stdevs)
    mvp_x = port_stdevs[idx_min]
    mvp_y = port_returns[idx_min]

    # ---------- 3) Handle the special case: mu_A == mu_B ----------
    tol = 1e-12  # tolerance for floating-point equality
    same_return = abs(mu_A - mu_B) < tol

    if same_return:
        #
        # The entire frontier is a horizontal line, so the only "efficient" point is the MVP.
        #
        # x_ef, y_ef => the single MVP point
        x_ef = [mvp_x]
        y_ef = [mvp_y]

        # x_inef, y_inef => everything else
        x_inef = np.delete(port_stdevs, idx_min)
        y_inef = np.delete(port_returns, idx_min)

    else:
    #    
    # ---------- 4) Determine which stock has the HIGHER return ----------
    # so we know which end of the curve is "efficient" (above the MVP).
    #   - If Stock A has higher return: w=1 => that’s the top end.
    #   - If Stock B has higher return: w=0 => that’s the top end.

    # w=0 => Stock B only
    ret_B, std_B = port_returns[0], port_stdevs[0]
    # w=1 => Stock A only
    ret_A, std_A = port_returns[-1], port_stdevs[-1]

    # Compare ret_A vs ret_B
    if ret_A > ret_B:
        # Stock A has the higher return -> The portion from the MVP to w=1 is efficient
        # The portion from w=0 to MVP is inefficient
        x_inef = port_stdevs[:idx_min+1]
        y_inef = port_returns[:idx_min+1]
        x_ef   = port_stdevs[idx_min:]
        y_ef   = port_returns[idx_min:]
    else:
        # Stock B has the higher return -> The portion from w=0 to the MVP is efficient
        # The portion from MVP to w=1 is inefficient
        x_ef = port_stdevs[:idx_min+1]
        y_ef = port_returns[:idx_min+1]
        x_inef   = port_stdevs[idx_min:]
        y_inef   = port_returns[idx_min:]

    # ---------- 5) Generate random portfolios for illustration ----------
    n_portfolios = 3000
    rand_weights = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []

    for w in rand_weights:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(p_return)
        rand_stdevs.append(np.sqrt(p_var))

    # ---------- 6) Plot ----------
    fig, ax = plt.subplots(figsize=(5, 3))

    # Random portfolios (gray dots)
    # ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # Efficient portion as solid red
    ax.plot(x_ef, y_ef, 'r-', linewidth=2, label='Efficient Frontier')

    # Inefficient portion as dashed red
    ax.plot(x_inef, y_inef, 'r--', label='Inefficient Portfolios')

    # Mark each stock individually (endpoints)
    # Remember: w=0 => Stock B, w=1 => Stock A
    ax.scatter(std_A, ret_A, s=50, marker='o', label='Stock A')
    ax.scatter(std_B, ret_B, s=50, marker='o', label='Stock B')

    # Mark the MVP with a black star
    ax.scatter(mvp_x, mvp_y, s=80, marker='*', color='black', label='Minimum-Variance Portfolio')

    ax.set_title("Two-Stock Frontier")
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    ax.legend(loc='best')
    plt.tight_layout()

    # Render the figure in Streamlit
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
