import numpy as np
import pandas as pd
import yfinance as yf

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches historical data for flexible tickers.
    Returns the data DataFrame AND a list of failed tickers.
    """
    stock_data = {}
    failed_tickers = []
    
    # Unique tickers only, convert to uppercase
    tickers = list(set([t.upper().strip() for t in tickers]))

    for stock in tickers:
        try:
            ticker_obj = yf.Ticker(stock)
            # Fetch slightly more data to ensure we have enough for returns calc
            history = ticker_obj.history(start=start_date, end=end_date)
            
            if history.empty:
                failed_tickers.append(stock)
            else:
                stock_data[stock] = history['Close']
                
        except Exception as e:
            print(f"Error fetching {stock}: {e}")
            failed_tickers.append(stock)
            
    if not stock_data:
        return pd.DataFrame(), failed_tickers
        
    return pd.DataFrame(stock_data), failed_tickers

def calculate_log_returns(stock_prices):
    """
    Calculates log returns from price data.
    """
    return np.log(stock_prices / stock_prices.shift(1)).dropna()

def perform_monte_carlo_simulation(log_returns, num_simulations, risk_free_rate=0.0):
    """
    Performs Monte Carlo simulation.
    """
    num_tickers = len(log_returns.columns)
    
    # Annualized parameters
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252
    
    # Generate random weights
    all_weights = np.random.random((num_simulations, num_tickers))
    all_weights = all_weights / np.sum(all_weights, axis=1)[:, np.newaxis]
    
    # Portfolio Returns
    port_returns = np.sum(all_weights * mean_returns.values, axis=1)
    
    # Portfolio Risk
    # Optimized calculation using Einstein summation for speed
    # (w @ cov @ w.T) for all simulations
    port_risks = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, cov_matrix, all_weights))
    
    sharpe_ratios = (port_returns - risk_free_rate) / port_risks
    
    # Create results DataFrame
    sim_data = {
        'Return': port_returns,
        'Risk': port_risks,
        'Sharpe Ratio': sharpe_ratios
    }
    
    # Add weights
    for idx, ticker in enumerate(log_returns.columns):
        sim_data[ticker] = all_weights[:, idx]
        
    results_df = pd.DataFrame(sim_data)
    
    max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
    optimal_portfolio = results_df.loc[max_sharpe_idx]
    
    return results_df, optimal_portfolio
