import streamlit as st
import numpy as np
import plotly.graph_objects as go


# Helper function to calculate frontier data
def calculate_frontier_data(mu_A, mu_B, sigma_A, sigma_B, rho):
    """Calculate frontier data for given parameters"""
    alphas = np.linspace(0, 1, 100)
    portfolio_returns = alphas * mu_A + (1 - alphas) * mu_B
    portfolio_stds = np.sqrt(
        alphas**2 * sigma_A**2 +
        (1 - alphas)**2 * sigma_B**2 +
        2 * alphas * (1 - alphas) * rho * sigma_A * sigma_B
    )
    
    # Compute Minimum Variance Portfolio
    denominator = sigma_A**2 + sigma_B**2 - 2 * rho * sigma_A * sigma_B
    if denominator == 0:
        w_star = sigma_B / (sigma_A + sigma_B)
    else:
        w_star = (sigma_B**2 - rho * sigma_A * sigma_B) / denominator
    
    w_star = max(0, min(w_star, 1))
    mvp_return = w_star * mu_A + (1 - w_star) * mu_B
    mvp_variance = w_star**2 * sigma_A**2 + (1 - w_star)**2 * sigma_B**2 + 2 * w_star * (1 - w_star) * rho * sigma_A * sigma_B
    mvp_std = np.sqrt(mvp_variance) if mvp_variance >= 0 else 0
    
    # Split into efficient and inefficient frontiers
    efficient_mask = portfolio_returns >= mvp_return
    efficient_returns = portfolio_returns[efficient_mask]
    efficient_stds = portfolio_stds[efficient_mask]
    inefficient_returns = portfolio_returns[~efficient_mask]
    inefficient_stds = portfolio_stds[~efficient_mask]
    
    return {
        'efficient_returns': efficient_returns,
        'efficient_stds': efficient_stds,
        'inefficient_returns': inefficient_returns,
        'inefficient_stds': inefficient_stds,
        'mvp_return': mvp_return,
        'mvp_std': mvp_std,
        'w_star': w_star,
        'mu_A': mu_A,
        'mu_B': mu_B,
        'sigma_A': sigma_A,
        'sigma_B': sigma_B
    }


# Set the page layout to wide and add a custom title/icon
st.set_page_config(
    page_title="Two-Asset Frontier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for saved frontiers
if 'saved_frontiers' not in st.session_state:
    st.session_state.saved_frontiers = []

# ── Session-state defaults for the six synced parameters ──
_DEFAULTS = {
    'mu_A_val': 8.9,
    'mu_B_val': 9.2,
    'sigma_A_val': 7.9,
    'sigma_B_val': 8.9,
    'last_rho': -0.5,
    'last_cov_pct': None,  # will be computed on first run
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sync callbacks: slider ↔ number_input ──
def _sync(src_key, dst_key):
    """Copy the value that the user just changed into the mirror widget."""
    st.session_state[dst_key] = st.session_state[src_key]

def _sync_to_state(src_key, state_key):
    """Copy widget value into a plain state key (no mirror widget)."""
    st.session_state[state_key] = st.session_state[src_key]


# Custom CSS for better styling (matching the previous apps)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
        font-size: 0.8rem;
        color: #64748B;
    }
    .stSlider label {
        font-weight: 500;
        color: #334155;
    }
    .plot-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0px 5px 5px 0px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        padding: 1rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    .metric-label {
        color: #64748B;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Custom header with logo/title
st.markdown('<h1 class="main-header">📈 Two-Asset Efficient Frontier</h1>', unsafe_allow_html=True)

# Add a description
with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown("""
    This tool visualizes the **Efficient Frontier** for a portfolio of two assets.
    
    - The **Efficient Frontier** shows all optimal portfolios that offer the highest expected return for a defined level of risk
    - The **Minimum Variance Portfolio (MVP)** is the portfolio with the lowest possible risk
    
    Adjust the sliders **or type a value directly into the number fields** to see how changes in returns, standard deviations, and correlation affect the frontier.
    """)

# Create columns for inputs and plot
col1, col2 = st.columns([1, 2])

with col1:
    # Card for asset inputs
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Asset Parameters</div>', unsafe_allow_html=True)
    
    # ── Expected Returns ──
    st.markdown("#### Expected Returns")

    # Stock A return
    sl_muA, ni_muA = st.columns([3, 1])
    with sl_muA:
        st.slider(
            'Expected Return of Stock A (%)',
            min_value=0.0, max_value=50.0, step=0.1,
            key='mu_A_slider',
            value=st.session_state.mu_A_val,
            on_change=_sync, args=('mu_A_slider', 'mu_A_number'),
            help="Annual expected return"
        )
    with ni_muA:
        st.number_input(
            'Stock A (%)',
            min_value=0.0, max_value=50.0, step=0.1,
            key='mu_A_number',
            value=st.session_state.mu_A_val,
            on_change=_sync, args=('mu_A_number', 'mu_A_slider'),
            label_visibility='collapsed'
        )
    mu_A = st.session_state.mu_A_slider
    st.session_state.mu_A_val = mu_A

    # Stock B return
    sl_muB, ni_muB = st.columns([3, 1])
    with sl_muB:
        st.slider(
            'Expected Return of Stock B (%)',
            min_value=0.0, max_value=50.0, step=0.1,
            key='mu_B_slider',
            value=st.session_state.mu_B_val,
            on_change=_sync, args=('mu_B_slider', 'mu_B_number'),
            help="Annual expected return"
        )
    with ni_muB:
        st.number_input(
            'Stock B (%)',
            min_value=0.0, max_value=50.0, step=0.1,
            key='mu_B_number',
            value=st.session_state.mu_B_val,
            on_change=_sync, args=('mu_B_number', 'mu_B_slider'),
            label_visibility='collapsed'
        )
    mu_B = st.session_state.mu_B_slider
    st.session_state.mu_B_val = mu_B

    # ── Risk Parameters ──
    st.markdown("#### Risk Parameters")

    # Stock A std dev
    sl_sA, ni_sA = st.columns([3, 1])
    with sl_sA:
        st.slider(
            'Standard Deviation of Stock A (%)',
            min_value=0.0, max_value=50.0, step=0.1,
            key='sigma_A_slider',
            value=st.session_state.sigma_A_val,
            on_change=_sync, args=('sigma_A_slider', 'sigma_A_number'),
            help="Annual standard deviation (volatility)"
        )
    with ni_sA:
        st.number_input(
            'Std A (%)',
            min_value=0.0, max_value=50.0, step=0.1,
            key='sigma_A_number',
            value=st.session_state.sigma_A_val,
            on_change=_sync, args=('sigma_A_number', 'sigma_A_slider'),
            label_visibility='collapsed'
        )
    sigma_A = st.session_state.sigma_A_slider
    st.session_state.sigma_A_val = sigma_A

    # Stock B std dev
    sl_sB, ni_sB = st.columns([3, 1])
    with sl_sB:
        st.slider(
            'Standard Deviation of Stock B (%)',
            min_value=0.0, max_value=50.0, step=0.1,
            key='sigma_B_slider',
            value=st.session_state.sigma_B_val,
            on_change=_sync, args=('sigma_B_slider', 'sigma_B_number'),
            help="Annual standard deviation (volatility)"
        )
    with ni_sB:
        st.number_input(
            'Std B (%)',
            min_value=0.0, max_value=50.0, step=0.1,
            key='sigma_B_number',
            value=st.session_state.sigma_B_val,
            on_change=_sync, args=('sigma_B_number', 'sigma_B_slider'),
            label_visibility='collapsed'
        )
    sigma_B = st.session_state.sigma_B_slider
    st.session_state.sigma_B_val = sigma_B

    # ── Relationship Between Assets ──
    st.markdown("#### Relationship Between Assets")
    
    # Toggle between correlation and covariance input
    input_mode = st.radio(
        "Input method:",
        options=["Correlation Coefficient", "Covariance"],
        horizontal=True,
        help="Choose whether to input correlation (ρ) or covariance directly"
    )
    
    # Calculate the valid covariance range based on current standard deviations
    sigma_A_decimal_temp = sigma_A / 100
    sigma_B_decimal_temp = sigma_B / 100
    max_cov = sigma_A_decimal_temp * sigma_B_decimal_temp
    min_cov = -max_cov
    
    # Convert to percentage for display (multiply by 10000 to show as percentage squared)
    max_cov_pct = max_cov * 10000
    min_cov_pct = min_cov * 10000
    
    # Initialise last_cov_pct on first run (depends on sigma values)
    if st.session_state.last_cov_pct is None:
        st.session_state.last_cov_pct = st.session_state.last_rho * max_cov * 10000
    
    if input_mode == "Correlation Coefficient":
        sl_rho, ni_rho = st.columns([3, 1])
        with sl_rho:
            st.slider(
                'Correlation Coefficient (ρ)',
                min_value=-1.0, max_value=1.0, step=0.01,
                key='rho_slider',
                value=st.session_state.last_rho,
                on_change=_sync, args=('rho_slider', 'rho_number'),
                help="Correlation between the two assets (-1 to 1)"
            )
        with ni_rho:
            st.number_input(
                'ρ',
                min_value=-1.0, max_value=1.0, step=0.01,
                key='rho_number',
                value=st.session_state.last_rho,
                on_change=_sync, args=('rho_number', 'rho_slider'),
                format="%.2f",
                label_visibility='collapsed'
            )
        rho = st.session_state.rho_slider
        # Update session state
        st.session_state.last_rho = rho
        # Calculate implied covariance
        covariance = rho * sigma_A_decimal_temp * sigma_B_decimal_temp
        # Update covariance in session state for when user switches back
        st.session_state.last_cov_pct = covariance * 10000
        st.info(f"**Implied Covariance:** {covariance*10000:.4f} (%²)")
    else:  # Covariance mode
        # Use the last correlation to calculate default covariance
        default_cov_pct = st.session_state.last_rho * max_cov * 10000
        # Clamp to valid range
        default_cov_pct = max(min_cov_pct, min(max_cov_pct, default_cov_pct))

        sl_cov, ni_cov = st.columns([3, 1])
        with sl_cov:
            st.slider(
                'Covariance (%²)',
                min_value=float(min_cov_pct),
                max_value=float(max_cov_pct),
                value=float(default_cov_pct),
                step=0.01,
                key='cov_slider',
                on_change=_sync, args=('cov_slider', 'cov_number'),
                help=f"Covariance between the two assets. Valid range: [{min_cov_pct:.2f}, {max_cov_pct:.2f}] %²"
            )
        with ni_cov:
            st.number_input(
                'Cov (%²)',
                min_value=float(min_cov_pct),
                max_value=float(max_cov_pct),
                value=float(default_cov_pct),
                step=0.01,
                key='cov_number',
                on_change=_sync, args=('cov_number', 'cov_slider'),
                format="%.4f",
                label_visibility='collapsed'
            )
        covariance_pct = st.session_state.cov_slider
        covariance = covariance_pct / 10000
        # Update session state
        st.session_state.last_cov_pct = covariance_pct
        
        # Calculate implied correlation
        if sigma_A_decimal_temp > 0 and sigma_B_decimal_temp > 0:
            rho = covariance / (sigma_A_decimal_temp * sigma_B_decimal_temp)
            # Clamp to [-1, 1] to handle numerical precision issues
            rho = max(-1.0, min(1.0, rho))
        else:
            rho = 0.0
        
        # Update correlation in session state for when user switches back
        st.session_state.last_rho = rho
        st.info(f"**Implied Correlation (ρ):** {rho:.4f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add buttons for saving and resetting frontiers (right after correlation/covariance input)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Frontier Overlay Controls</div>', unsafe_allow_html=True)
    
    # Convert sliders to decimal form for calculations (need these for saving)
    mu_A_decimal = mu_A / 100
    mu_B_decimal = mu_B / 100
    sigma_A_decimal = sigma_A / 100
    sigma_B_decimal = sigma_B / 100
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button('💾 Save Current Frontier', use_container_width=True):
            # Save current frontier configuration (in decimal form)
            frontier_data = {
                'mu_A': mu_A_decimal,
                'mu_B': mu_B_decimal,
                'sigma_A': sigma_A_decimal,
                'sigma_B': sigma_B_decimal,
                'rho': rho,
                'label': f'Frontier {len(st.session_state.saved_frontiers) + 1}'
            }
            st.session_state.saved_frontiers.append(frontier_data)
            st.success(f'✅ Saved as Frontier {len(st.session_state.saved_frontiers)}')
    
    with col_btn2:
        if st.button('🔄 Reset All Frontiers', use_container_width=True):
            st.session_state.saved_frontiers = []
            st.info('All saved frontiers cleared')
    
    if len(st.session_state.saved_frontiers) > 0:
        st.markdown(f"**{len(st.session_state.saved_frontiers)} frontier(s) saved**")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Information box for correlation and covariance (now appears after the buttons)
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    if input_mode == "Correlation Coefficient":
        covariance = rho * sigma_A_decimal_temp * sigma_B_decimal_temp
    st.markdown(f"""
    **Current Values:**
    - Correlation (ρ): {rho:.4f}
    - Covariance: {covariance*10000:.4f} (%²)
    
    **Valid Covariance Range:** [{min_cov_pct:.2f}, {max_cov_pct:.2f}] %²
    
    **Correlation interpretation:**
    - Perfect negative: -1.0
    - No correlation: 0.0
    - Perfect positive: 1.0
    
    Diversification benefits are strongest when correlation is negative or low.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Use decimal versions for the rest of calculations
mu_A = mu_A_decimal
mu_B = mu_B_decimal
sigma_A = sigma_A_decimal
sigma_B = sigma_B_decimal

# Generate parametric minimum-variance frontier
alphas = np.linspace(0, 1, 100)
portfolio_returns = alphas * mu_A + (1 - alphas) * mu_B
portfolio_stds = np.sqrt(
    alphas**2 * sigma_A**2 +
    (1 - alphas)**2 * sigma_B**2 +
    2 * alphas * (1 - alphas) * rho * sigma_A * sigma_B
)

# Compute Minimum Variance Portfolio (MVP)
denominator = sigma_A**2 + sigma_B**2 - 2 * rho * sigma_A * sigma_B

# Handle division by zero
if denominator == 0:
    w_star = sigma_B / (sigma_A + sigma_B)  # Special handling for rho = -1
else:
    w_star = (sigma_B**2 - rho * sigma_A * sigma_B) / denominator

w_star = max(0, min(w_star, 1))  # Ensure no short sales

mvp_return = w_star * mu_A + (1 - w_star) * mu_B

# Correctly calculate MVP standard deviation
mvp_variance = w_star**2 * sigma_A**2 + (1 - w_star)**2 * sigma_B**2 + 2 * w_star * (1 - w_star) * rho * sigma_A * sigma_B

# Check if variance is non-negative before calculating standard deviation
if mvp_variance >= 0:
    mvp_std = np.sqrt(mvp_variance)
else:
    mvp_std = 0  # Set to 0 if variance is negative (theoretical minimum variance portfolio)

# Special case handling: If returns are equal, MVP is the only efficient portfolio
if mu_A == mu_B:
    # Create the plotly figure for equal returns case
    with col2:
        st.markdown('<div class="card plot-container">', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Define colors for saved frontiers
        saved_colors = ['#9CA3AF', '#6B7280', '#4B5563', '#374151', '#1F2937']
        
        # Add saved frontiers first (in background)
        for idx, saved_frontier in enumerate(st.session_state.saved_frontiers):
            saved_data = calculate_frontier_data(
                saved_frontier['mu_A'],
                saved_frontier['mu_B'],
                saved_frontier['sigma_A'],
                saved_frontier['sigma_B'],
                saved_frontier['rho']
            )
            color = saved_colors[idx % len(saved_colors)]
            
            # Add saved efficient frontier
            fig.add_trace(go.Scatter(
                x=saved_data['efficient_stds'] * 100,
                y=saved_data['efficient_returns'] * 100,
                mode='lines',
                name=saved_frontier['label'],
                line=dict(color=color, width=2),
                opacity=0.6,
                hovertemplate=f"{saved_frontier['label']}<br>Risk: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<extra></extra>"
            ))
            
            # Add saved inefficient frontier
            fig.add_trace(go.Scatter(
                x=saved_data['inefficient_stds'] * 100,
                y=saved_data['inefficient_returns'] * 100,
                mode='lines',
                name=f"{saved_frontier['label']} (Inefficient)",
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.4,
                showlegend=False,
                hovertemplate=f"{saved_frontier['label']}<br>Risk: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<extra></extra>"
            ))
        
        # Add Stock A and B points
        fig.add_trace(go.Scatter(
            x=[sigma_A * 100],
            y=[mu_A * 100],
            mode='markers',
            marker=dict(size=12, color='#10B981', symbol='square'),
            name='Stock A',
            hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Asset: Stock A<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[sigma_B * 100],
            y=[mu_B * 100],
            mode='markers',
            marker=dict(size=12, color='#F97316', symbol='square'),
            name='Stock B',
            hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Asset: Stock B<extra></extra>'
        ))
        
        # Add MVP point
        fig.add_trace(go.Scatter(
            x=[mvp_std * 100],
            y=[mvp_return * 100],
            mode='markers',
            marker=dict(size=14, color='#EF4444', symbol='star'),
            name='Minimum Variance Portfolio',
            hovertemplate=f'Risk: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<br>Stock A: {w_star*100:.1f}%<br>Stock B: {(1-w_star)*100:.1f}%<extra></extra>'
        ))
        
        # Customize the layout
        max_std = max(sigma_A, sigma_B) * 100 + 1
        min_return = min(mu_A, mu_B) * 100 - 1
        max_return = max(mu_A, mu_B) * 100 + 1
        
        fig.update_layout(
            title=dict(
                text="Minimum Variance Portfolio",
                font=dict(size=24, family="Arial, sans-serif", color="#1E3A8A"),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Standard Deviation (%)",
                tickformat='.1f',
                gridcolor='rgba(230, 230, 230, 0.8)',
                range=[0, max_std]
            ),
            yaxis=dict(
                title="Expected Return (%)",
                tickformat='.1f',
                gridcolor='rgba(230, 230, 230, 0.8)',
                range=[min_return, max_return]
            ),
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            height=600,
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results box
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Portfolio Results</div>', unsafe_allow_html=True)
        
        # Create a grid for metrics
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{mvp_std*100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Minimum Standard Deviation</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{mvp_return*100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">MVP Return</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_c:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{w_star*100:.1f}% / {(1-w_star)*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">MVP Weights (Stock A/B)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Split into efficient and inefficient frontiers
    efficient_mask = portfolio_returns >= mvp_return  # Keep only points above or equal to MVP's return
    efficient_returns = portfolio_returns[efficient_mask]
    efficient_stds = portfolio_stds[efficient_mask]
    inefficient_returns = portfolio_returns[~efficient_mask]
    inefficient_stds = portfolio_stds[~efficient_mask]
    
    # Create the plotly figure for normal case
    with col2:
        st.markdown('<div class="card plot-container">', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Define colors for saved frontiers (cycle through these)
        saved_colors = ['#9CA3AF', '#6B7280', '#4B5563', '#374151', '#1F2937']
        
        # Add saved frontiers first (in background)
        for idx, saved_frontier in enumerate(st.session_state.saved_frontiers):
            saved_data = calculate_frontier_data(
                saved_frontier['mu_A'],
                saved_frontier['mu_B'],
                saved_frontier['sigma_A'],
                saved_frontier['sigma_B'],
                saved_frontier['rho']
            )
            color = saved_colors[idx % len(saved_colors)]
            
            # Add saved efficient frontier
            fig.add_trace(go.Scatter(
                x=saved_data['efficient_stds'] * 100,
                y=saved_data['efficient_returns'] * 100,
                mode='lines',
                name=saved_frontier['label'],
                line=dict(color=color, width=2),
                opacity=0.6,
                hovertemplate=f"{saved_frontier['label']}<br>Risk: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<extra></extra>"
            ))
            
            # Add saved inefficient frontier
            fig.add_trace(go.Scatter(
                x=saved_data['inefficient_stds'] * 100,
                y=saved_data['inefficient_returns'] * 100,
                mode='lines',
                name=f"{saved_frontier['label']} (Inefficient)",
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.4,
                showlegend=False,
                hovertemplate=f"{saved_frontier['label']}<br>Risk: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<extra></extra>"
            ))
        
        # Add current efficient frontier (highlighted)
        fig.add_trace(go.Scatter(
            x=efficient_stds * 100,
            y=efficient_returns * 100,
            mode='lines',
            name='Current Efficient Frontier',
            line=dict(color='#2563EB', width=4),
            hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add current inefficient frontier
        fig.add_trace(go.Scatter(
            x=inefficient_stds * 100,
            y=inefficient_returns * 100,
            mode='lines',
            name='Current Inefficient Frontier',
            line=dict(color='#2563EB', width=4, dash='dash'),
            hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add Stock A and B points
        fig.add_trace(go.Scatter(
            x=[sigma_A * 100],
            y=[mu_A * 100],
            mode='markers',
            marker=dict(size=12, color='#10B981', symbol='square'),
            name='Stock A',
            hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Asset: Stock A<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[sigma_B * 100],
            y=[mu_B * 100],
            mode='markers',
            marker=dict(size=12, color='#F97316', symbol='square'),
            name='Stock B',
            hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Asset: Stock B<extra></extra>'
        ))
        
        # Add MVP point
        fig.add_trace(go.Scatter(
            x=[mvp_std * 100],
            y=[mvp_return * 100],
            mode='markers',
            marker=dict(size=14, color='#EF4444', symbol='star'),
            name='Minimum Variance Portfolio',
            hovertemplate=f'Risk: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<br>Stock A: {w_star*100:.1f}%<br>Stock B: {(1-w_star)*100:.1f}%<extra></extra>'
        ))
        
        # Set x-axis limits with default value for empty lists
        max_efficient_std = max(efficient_stds, default=0)
        max_inefficient_std = max(inefficient_stds, default=0)
        
        # Customize the layout
        fig.update_layout(
            title=dict(
                text="Efficient Frontier with Minimum-Variance Portfolio (MVP)",
                font=dict(size=24, family="Arial, sans-serif", color="#1E3A8A"),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Standard Deviation (%)",
                tickformat='.1f',
                gridcolor='rgba(230, 230, 230, 0.8)',
                range=[0, max(max_efficient_std, max_inefficient_std) * 100 + 1]
            ),
            yaxis=dict(
                title="Expected Return (%)",
                tickformat='.1f',
                gridcolor='rgba(230, 230, 230, 0.8)',
                range=[min(min(efficient_returns), min(inefficient_returns, default=mu_A)) * 100 - 1, 
                      max(max(efficient_returns), max(inefficient_returns, default=mu_B)) * 100 + 1]
            ),
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            height=600,
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results box
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Portfolio Results</div>', unsafe_allow_html=True)
        
        # Create a grid for metrics
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{mvp_std*100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Minimum Standard Deviation</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{mvp_return*100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">MVP Return</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_c:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{w_star*100:.1f}% / {(1-w_star)*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">MVP Weights (Stock A/B)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Add educational content
with st.expander("📘 Understanding the Efficient Frontier", expanded=False):
    st.markdown("""
    ### Key Concepts
    
    **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a defined level of risk.
    
    **Minimum Variance Portfolio (MVP)**: The portfolio with the lowest possible risk, regardless of return.
    
    **Correlation vs. Covariance**:
    - **Correlation (ρ)**: Standardized measure of linear relationship, ranges from -1 to +1
    - **Covariance**: Actual measure of how returns move together, units are (%²)
    - **Relationship**: Cov(A,B) = ρ × σ_A × σ_B
    
    **Correlation/Covariance Effects**:
    - Negative correlation/covariance: Strong diversification benefits
    - Zero correlation/covariance: Good diversification 
    - Positive correlation/covariance: Limited diversification benefits
    
    ### Portfolio Optimization (using covariance)
    
    The weight of asset A in the minimum variance portfolio is given by:
    
    $$x_A = \\frac{\\sigma^2_B - Cov(A,B)}{\\sigma^2_A + \\sigma^2_B - 2 \\cdot Cov(A,B)}$$
    
    where:
    - $\\sigma^2_A$, $\\sigma^2_B$ = variances of assets A and B
    - $Cov(A,B)$ = covariance between assets A and B
    
    **Alternative formula (using correlation)**:
    
    $$x_A = \\frac{\\sigma^2_B - \\rho \\cdot \\sigma_A \\cdot \\sigma_B}{\\sigma^2_A + \\sigma^2_B - 2 \\cdot \\rho \\cdot \\sigma_A \\cdot \\sigma_B}$$
    """)
    
    # Display additional formulas
    st.markdown("""
    ### Portfolio Standard Deviation
    
    **Using covariance**:
    
    $$\\sigma_p = \\sqrt{x_A^2 \\sigma_A^2 + x_B^2 \\sigma_B^2 + 2x_A x_B \\cdot Cov(A,B)}$$
    
    **Using correlation**:
    
    $$\\sigma_p = \\sqrt{x_A^2 \\sigma_A^2 + x_B^2 \\sigma_B^2 + 2x_A x_B\\rho\\sigma_A\\sigma_B}$$
    
    ### Portfolio Return
    
    The expected return of a two-asset portfolio is:
    
    $$E(R_p) = x_A \\cdot E(R_A) + x_B \\cdot E(R_B)$$

    where:

    $$x_A + x_B = 1$$
    """)


# Footer
st.markdown('<div class="footer">Two-Asset Efficient Frontier Visualizer | Developed by Prof. Marc Goergen with the help of ChatGPT, Perplexity and Claude</div>', unsafe_allow_html=True)
