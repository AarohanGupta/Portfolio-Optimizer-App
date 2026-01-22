import streamlit as st
import pandas as pd
import plotly.express as px
import utils
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Pro Portfolio Optimizer", layout="wide", page_icon="📈")

st.title("📈 Pro Portfolio Optimizer")
st.markdown("""
**Optimize any portfolio globally.** Enter ticker symbols below (e.g., `AAPL, MSFT` for US, `RELIANCE.NS, TCS.NS` for India, `BTC-USD` for Crypto).
""")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# Robust Input Method: Text Area
default_input = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ITC.NS"
ticker_input = st.sidebar.text_area(
    "Enter Tickers (comma-separated)", 
    value=default_input,
    height=100,
    help="You can enter any ticker supported by Yahoo Finance."
)

# Parse inputs
raw_tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = col2.date_input("End Date", value=datetime.today())

num_simulations = st.sidebar.slider("Simulations", 1000, 50000, 10000, step=1000)
run_btn = st.sidebar.button("Run Optimization", type="primary")

# --- Main Logic ---
if run_btn:
    if len(raw_tickers) < 2:
        st.error("⚠️ Please enter at least 2 distinct tickers to form a portfolio.")
    else:
        with st.spinner("Fetching global market data..."):
            # 1. Fetch Data (with error handling)
            stock_prices, failed_tickers = utils.fetch_stock_data(raw_tickers, start_date, end_date)
            
            # 2. Handle Failed Tickers
            if failed_tickers:
                st.warning(f"⚠️ Could not find data for: {', '.join(failed_tickers)}. Check spelling or delisted status.")
            
            # 3. Check if we have enough valid data remaining
            if stock_prices.empty or stock_prices.shape[1] < 2:
                st.error("❌ Not enough valid stocks found to proceed. Please check your tickers.")
            else:
                st.success(f"✅ Successfully loaded data for {len(stock_prices.columns)} assets.")
                
                # 4. Processing
                log_returns = utils.calculate_log_returns(stock_prices)
                sim_results, optimal_port = utils.perform_monte_carlo_simulation(log_returns, num_simulations)
                
                # 5. Visualizations
                st.subheader("🏆 Optimal Portfolio Allocation")
                
                # Metrics Row
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected Return", f"{optimal_port['Return']:.2%}")
                c2.metric("Risk (Volatility)", f"{optimal_port['Risk']:.2%}")
                c3.metric("Sharpe Ratio", f"{optimal_port['Sharpe Ratio']:.2f}")
                
                st.divider()
                
                col_chart, col_weights = st.columns([2, 1])
                
                # Efficient Frontier Plot
                with col_chart:
                    st.subheader("Efficient Frontier")
                    fig = px.scatter(
                        sim_results, x="Risk", y="Return", color="Sharpe Ratio",
                        color_continuous_scale="Viridis",
                        title="Monte Carlo Simulation",
                        hover_data=stock_prices.columns
                    )
                    # Add optimal point
                    fig.add_scatter(
                        x=[optimal_port['Risk']], y=[optimal_port['Return']],
                        mode='markers', marker=dict(size=15, color='red', symbol='star'),
                        name='Max Sharpe'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Weights Table & Pie Chart
                with col_weights:
                    st.subheader("Weights")
                    # Clean weights data
                    assets = stock_prices.columns
                    weights = optimal_port[assets]
                    
                    # Filter out tiny weights (< 1%) for cleaner view
                    weights_df = pd.DataFrame({'Asset': assets, 'Weight': weights})
                    weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
                    
                    # Display Table
                    st.dataframe(
                        weights_df.style.format({"Weight": "{:.2%}"}), 
                        use_container_width=True, 
                        hide_index=True
                    )
                    
                    # Pie Chart
                    fig_pie = px.pie(weights_df, values='Weight', names='Asset', hole=0.4)
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie, use_container_width=True)

elif not run_btn:
    st.info("👈 Enter tickers in the sidebar and click Run.")
