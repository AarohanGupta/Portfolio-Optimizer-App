import streamlit as st
import pandas as pd
import plotly.express as px
import utils
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

st.title("📈 Monte Carlo Portfolio Optimization")
st.markdown("Interact with the sidebar to select stocks, simulation parameters, and date ranges.")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# Default values based on your notebook
default_tickers = ["RELIANCE.NS", "TCS.NS", "HINDUNILVR.NS", "HDFCBANK.NS", "ITC.NS", "LT.NS", "INFY.NS"]
# Extended list for demo purposes
available_tickers = default_tickers + ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "NVDA"]

selected_tickers = st.sidebar.multiselect(
    "Select Stocks", 
    options=available_tickers, 
    default=default_tickers
)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=datetime(2018, 10, 1))
end_date = col2.date_input("End Date", value=datetime(2023, 10, 1))

num_simulations = st.sidebar.slider(
    "Number of Simulations", 
    min_value=1000, 
    max_value=20000, 
    value=10000, 
    step=1000,
    help="Higher numbers give more accurate results but take longer to run."
)

run_btn = st.sidebar.button("Run Optimization", type="primary")

# --- Main Application Logic ---
if run_btn:
    if not selected_tickers or len(selected_tickers) < 2:
        st.error("Please select at least two stocks to optimize a portfolio.")
    else:
        try:
            with st.spinner("Fetching data and simulating portfolios..."):
                # 1. Fetch Data
                stock_prices = utils.fetch_stock_data(selected_tickers, start_date, end_date)
                
                if stock_prices.empty:
                    st.error("No data found for the selected tickers. Please check the symbols or date range.")
                else:
                    # 2. Process Data
                    log_returns = utils.calculate_log_returns(stock_prices)
                    
                    # 3. Perform Simulation
                    sim_results, optimal_port = utils.perform_monte_carlo_simulation(log_returns, num_simulations)
                    
                    # --- Display Results ---
                    
                    # Top KPI Metrics
                    st.subheader("🏆 Optimal Portfolio Metrics (Max Sharpe Ratio)")
                    kpi1, kpi2, kpi3 = st.columns(3)
                    
                    kpi1.metric("Expected Annual Return", f"{optimal_port['Return']:.2%}")
                    kpi2.metric("Annual Volatility (Risk)", f"{optimal_port['Risk']:.2%}")
                    kpi3.metric("Sharpe Ratio", f"{optimal_port['Sharpe Ratio']:.2f}")
                    
                    st.divider()
                    
                    # Layout: Weights Table & Charts
                    col_charts, col_table = st.columns([2, 1])
                    
                    # Extract just the weights for the optimal portfolio
                    weights_series = optimal_port[selected_tickers]
                    weights_df = pd.DataFrame({
                        'Stock': weights_series.index, 
                        'Weight': weights_series.values
                    })
                    
                    with col_charts:
                        st.subheader("📊 Efficient Frontier")
                        
                        # Plotly Scatter Plot
                        fig = px.scatter(
                            sim_results, 
                            x="Risk", 
                            y="Return", 
                            color="Sharpe Ratio",
                            title="Risk vs Return (Colored by Sharpe Ratio)",
                            hover_data=selected_tickers,
                            color_continuous_scale="Viridis",
                            labels={"Risk": "Annualized Volatility", "Return": "Annualized Return"}
                        )
                        
                        # Highlight the Optimal Portfolio
                        fig.add_scatter(
                            x=[optimal_port['Risk']], 
                            y=[optimal_port['Return']], 
                            mode='markers', 
                            marker=dict(color='red', size=15, symbol='star'),
                            name='Optimal Portfolio'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                    with col_table:
                        st.subheader("📦 Optimal Weights")
                        
                        # Formatting weights for display
                        display_weights = weights_df.copy()
                        display_weights['Weight'] = display_weights['Weight'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(display_weights, hide_index=True, use_container_width=True)
                        
                        # Pie chart for weights
                        fig_pie = px.pie(weights_df, values='Weight', names='Stock', title='Asset Allocation')
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Select your stocks and parameters from the sidebar, then click **Run Optimization** to begin.")