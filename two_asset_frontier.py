import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    """
    If mu_A == mu_B (within tolerance):
       * entire line is labeled 'Inefficient' (dashed)
       * the MVP (lowest std dev) is a single red dot for 'Efficient Frontier'
    Otherwise:
       * standard 2-stock logic (one portion is solid 'efficient', the other dashed 'inefficient').
    Legend placed outside, wide figure, smaller font, random portfolios in gray for illustration.
    """

    # 1) Parametric Frontier: w in [0,1]
    cov_AB = corr_AB * sigma_A * sigma_B
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    frontier_returns = []
    frontier_stdevs  = []

    for w in weights:
        r = w * mu_A + (1 - w) * mu_B
        v = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        frontier_returns.append(r)
        frontier_stdevs.append(np.sqrt(v))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs  = np.array(frontier_stdevs)

    # 2) Minimum-Variance Portfolio (MVP)
    idx_min = np.argmin(frontier_stdevs)
    mvp_x   = frontier_stdevs[idx_min]
    mvp_y   = frontier_returns[idx_min]

    # 3) Random portfolios
    n_portfolios = 3000
    rand_w = np.random.rand(n_portfolios)
    rand_returns = []
    rand_stdevs  = []
    for w in rand_w:
        rp_return = w * mu_A + (1 - w) * mu_B
        rp_var    = (w**2)*(sigma_A**2) + ((1-w)**2)*(sigma_B**2) + 2*w*(1-w)*cov_AB
        rand_returns.append(rp_return)
        rand_stdevs.append(np.sqrt(rp_var))

    # 4) Check if returns are effectively the same
    tol = 1e-12
    same_return = (abs(mu_A - mu_B) < tol)

    if same_return:
        # ========== CASE: Stocks have identical returns => only MVP is truly efficient ==========

        # The entire parametric line => 'inefficient' except MVP
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False
        inef_x = frontier_stdevs[mask]
        inef_y = frontier_returns[mask]

        # The MVP => single red dot labeled 'Efficient Frontier'
        ef
