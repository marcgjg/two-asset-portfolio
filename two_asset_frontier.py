import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

try:
    # Set layout to wide
    st.set_page_config(layout="wide")

    # Define sliders for inputs
    col1, col2 = st.columns(2)
    with col1:
        mu_A = st.slider('Expected Return of Stock A (%)', min_value=0.0, max_value=50.0, value=7.0, step=0.1)
    with col2:
        mu_B = st.slider('Expected Return of Stock B (%)', min_value=0.0, max_value=50.0, value=18.0, step=0.1)
    
    col3, col4 = st.columns(2)
    with col3:
        sigma_A = st.slider('Standard Deviation of Stock A (%)', min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    with col4:
        sigma_B = st.slider('Standard Deviation of Stock B (%)', min_value=0.0, max_value=50.0, value=30.0, step=0.1)
    
    rho = st.slider('Correlation Coefficient', min_value=-1.0, max_value=1.0, value=0.08, step=0.01)

    # Convert sliders back to decimal form for calculations
    mu_A = mu_A / 100
    mu_B = mu_B / 100
    sigma_A = sigma_A / 100
    sigma_B = sigma_B / 100

    # Compute Minimum Variance Portfolio if returns are equal
    if mu_A == mu_B:
        w_star = (sigma_B**2 - rho * sigma_A * sigma_B) / (sigma_A**2 + sigma_B**2 - 2 * rho * sigma_A * sigma_B)
        # Ensure no short sales
        w_star = max(0, min(w_star, 1))
        portfolio_return = mu_A
        portfolio_std = np.sqrt(w_star**2 * sigma_A**2 + (1 - w_star)**2 * sigma_B**2 + 2 * w_star * (1 - w_star) * rho * sigma_A * sigma_B)

        # Plot MVP
        col5, col6 = st.columns([3, 1])  # Adjust column widths
        with col5:
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.scatter(sigma_A*100, mu_A*100, color='blue', label='Stock A')  # Convert back to percentage for plotting
            ax.scatter(sigma_B*100, mu_B*100, color='green', label='Stock B')
            ax.scatter(portfolio_std*100, portfolio_return*100, color='red', label=f'MVP ({portfolio_std*100:.2f}, {portfolio_return*100:.2f})')
            ax.scatter(portfolio_std*100, portfolio_return*100, marker='*', color='black', s=200)
            ax.set_xlabel('Standard Deviation')
            ax.set_ylabel('Expected Return')
            ax.set_title('Minimum Variance Portfolio')
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            
            # Manually set tick labels without percentage sign
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.1f}".format(x)))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.1f}".format(x)))
            st.pyplot(fig)

    else:
        # Generate parametric efficient frontier
        alphas = np.linspace(0, 1, 100)
        portfolio_returns = alphas * mu_A + (1 - alphas) * mu_B
        portfolio_stds = np.sqrt(alphas**2 * sigma_A**2 + (1 - alphas)**2 * sigma_B**2 + 2 * alphas * (1 - alphas) * rho * sigma_A * sigma_B)

        # Find MVP within the efficient frontier
        mvp_idx = np.argmin(portfolio_stds)
        mvp_return = portfolio_returns[mvp_idx]
        mvp_std = portfolio_stds[mvp_idx]

        # Split into efficient and inefficient parts
        max_return_idx = np.argmax(portfolio_returns)
        if mvp_idx < max_return_idx:
            efficient_returns = portfolio_returns[mvp_idx:max_return_idx+1]
            efficient_stds = portfolio_stds[mvp_idx:max_return_idx+1]
            inefficient_returns = portfolio_returns[:mvp_idx]
            inefficient_stds = portfolio_stds[:mvp_idx]
        else:
            efficient_returns = portfolio_returns[max_return_idx:mvp_idx+1]
            efficient_stds = portfolio_stds[max_return_idx:mvp_idx+1]
            inefficient_returns = portfolio_returns[mvp_idx+1:]
            inefficient_stds = portfolio_stds[mvp_idx+1:]

        # Plot efficient frontier
        col5, col6 = st.columns([3, 1])  # Adjust column widths
        with col5:
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.scatter(sigma_A*100, mu_A*100, color='blue', label='Stock A')  # Convert back to percentage for plotting
            ax.scatter(sigma_B*100, mu_B*100, color='green', label='Stock B')
            ax.plot(efficient_stds*100, efficient_returns*100, color='red', label='Efficient Frontier')  # Convert back to percentage for plotting
            ax.plot(inefficient_stds*100, inefficient_returns*100, color='red', linestyle='--', label='Inefficient Frontier')
            ax.scatter(mvp_std*100, mvp_return*100, marker='*', color='black', s=200, label=f'MVP ({mvp_std*100:.2f}, {mvp_return*100:.2f})')
            ax.scatter(mvp_std*100, mvp_return*100, color='red')
            
            # Optionally include random portfolios
            if st.checkbox('Include Random Portfolios'):
                random_alphas = np.random.uniform(0, 1, size=100)
                random_returns = random_alphas * mu_A + (1 - random_alphas) * mu_B
                random_stds = np.sqrt(random_alphas**2 * sigma_A**2 + (1 - random_alphas)**2 * sigma_B**2 + 2 * random_alphas * (1 - random_alphas) * rho * sigma_A * sigma_B)
                ax.scatter(random_stds*100, random_returns*100, color='gray', alpha=0.5)  # Convert back to percentage for plotting

            ax.set_xlabel('Standard Deviation')
            ax.set_ylabel('Expected Return')
            ax.set_title('Efficient Frontier')
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            
            # Manually set tick labels without percentage sign
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.1f}".format(x)))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.1f}".format(x)))
            st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
