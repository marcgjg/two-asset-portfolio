import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

def plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB):
    # 1) param frontier
    cov_AB = corr_AB * sigma_A * sigma_B
    w = np.linspace(0, 1, 200)
    frontier_returns = []
    frontier_stdevs  = []

    for x in w:
        r = x*mu_A + (1 - x)*mu_B
        v = (x**2)*(sigma_A**2) + ((1 - x)**2)*(sigma_B**2) + 2*x*(1 - x)*cov_AB
        frontier_returns.append(r)
        frontier_stdevs.append(np.sqrt(v))

    frontier_returns = np.array(frontier_returns)
    frontier_stdevs  = np.array(frontier_stdevs)

    # 2) find MVP
    idx_min = np.argmin(frontier_stdevs)
    mvp_x = frontier_stdevs[idx_min]
    mvp_y = frontier_returns[idx_min]

    # 3) check same return
    tol = 1e-12
    same_return = (abs(mu_A - mu_B) < tol)

    fig, ax = plt.subplots(figsize=(6, 4))

    if same_return:
        # entire line => dashed
        mask = np.ones_like(frontier_stdevs, dtype=bool)
        mask[idx_min] = False
        x_inef = frontier_stdevs[mask]
        y_inef = frontier_returns[mask]

        # MVP alone => single red dot
        x_ef = [mvp_x]
        y_ef = [mvp_y]

        ax.plot(x_inef, y_inef, 'r--', label='Inefficient')
        ax.plot(x_ef, y_ef, 'ro', label='Efficient Frontier (single point)')

    else:
        # normal logic
        if mu_A > mu_B:
            x_inef = frontier_stdevs[:idx_min+1]
            y_inef = frontier_returns[:idx_min+1]
            x_ef   = frontier_stdevs[idx_min:]
            y_ef   = frontier_returns[idx_min:]
        else:
            x_ef   = frontier_stdevs[:idx_min+1]
            y_ef   = frontier_returns[:idx_min+1]
            x_inef = frontier_stdevs[idx_min:]
            y_inef = frontier_returns[idx_min:]

        ax.plot(x_inef, y_inef, 'r--', label='Inefficient')
        ax.plot(x_ef, y_ef, 'r-', label='Efficient Frontier')

    ax.plot(mvp_x, mvp_y, 'k*', label='MVP')
    ax.set_xlabel('Std Dev')
    ax.set_ylabel('Return')

    ax.set_title(f"muA={mu_A}, muB={mu_B}, sigmaA={sigma_A}, sigmaB={sigma_B}, corr={corr_AB}")
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Minimal Single-Point Frontier if Returns Match")

    mu_A = st.slider("Stock A Return", 0.00, 0.20, 0.03, 0.01)
    mu_B = st.slider("Stock B Return", 0.00, 0.20, 0.03, 0.01)
    sigma_A = st.slider("Stock A Std Dev", 0.01, 0.40, 0.12, 0.01)
    sigma_B = st.slider("Stock B Std Dev", 0.01, 0.40, 0.15, 0.01)
    corr_AB = st.slider("Correlation", -1.0, 1.0, -0.70, 0.05)

    plot_two_stock_frontier(mu_A, mu_B, sigma_A, sigma_B, corr_AB)

if __name__ == "__main__":
    main()
