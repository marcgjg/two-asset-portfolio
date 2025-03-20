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

    # ---------- 3) Determine which stock has the HIGHER return ----------
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

    # ---------- 4) Generate random portfolios for illustration ----------
    n_portfolios = 3000
    rand_weights = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []

    for w in rand_weights:
        p_return = w * mu_A + (1 - w) * mu_B
        p_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(p_return)
        rand_stdevs.append(np.sqrt(p_var))

    # ---------- 5) Plot ----------
    fig, ax = plt.subplots(figsize=(5, 3))

    # Random portfolios (gray dots)
    ax.scatter(rand_stdevs, rand_returns, alpha=0.2, s=10, color='gray', label='Random Portfolios')

    # Inefficient portion as dashed red
    ax.plot(x_inef, y_inef, 'r--', label='Inefficient')

    # Efficient portion as solid red
    ax.plot(x_ef, y_ef, 'r-', linewidth=2, label='Efficient Frontier')

    # Mark each stock individually (endpoints)
    # Remember: w=0 => Stock B, w=1 => Stock A
    ax.scatter(std_B, ret_B, s=50, marker='o', label='St
