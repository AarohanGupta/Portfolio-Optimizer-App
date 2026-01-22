import numpy as np
import pandas as pd
import yfinance as yf

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches historical closing prices for a list of tickers using yfinance.
    
    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str or datetime): Start date for data fetching.
        end_date (str or datetime): End date for data fetching.
        
    Returns:
        pd.DataFrame: DataFrame containing closing prices for the tickers.
    """
    stock_data = {}
    
    # Using a loop to handle potential errors per ticker gracefully, 
    # though yf.download can also handle lists.
    for stock in tickers:
        try:
            ticker_obj = yf.Ticker(stock)
            history = ticker_obj.history(start=start_date, end=end_date)
            
            if not history.empty:
                stock_data[stock] = history['Close']
        except Exception as e:
            print(f"Error fetching {stock}: {e}")
            
    if not stock_data:
        return pd.DataFrame()
        
    return pd.DataFrame(stock_data)

def calculate_log_returns(stock_prices):
    """
    Calculates log returns from price data.
    
    Args:
        stock_prices (pd.DataFrame): DataFrame of stock prices.
        
    Returns:
        pd.DataFrame: DataFrame of log returns.
    """
    # Log returns: ln(Price_t / Price_t-1)
    return np.log(stock_prices / stock_prices.shift(1)).dropna()

def perform_monte_carlo_simulation(log_returns, num_simulations, risk_free_rate=0.0):
    """
    Performs Monte Carlo simulation to generate random long-only portfolios.
    
    Args:
        log_returns (pd.DataFrame): Historical log returns.
        num_simulations (int): Number of portfolios to simulate.
        risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation.
        
    Returns:
        tuple: (simulation_results_df, optimal_portfolio_series)
    """
    num_tickers = len(log_returns.columns)
    
    # Annualized mean returns and covariance matrix (252 trading days)
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252
    
    # Generate random weights for all simulations at once
    # Shape: (num_simulations, num_tickers)
    all_weights = np.random.random((num_simulations, num_tickers))
    
    # Normalize weights so they sum to 1 (Long-only constraint)
    all_weights = all_weights / np.sum(all_weights, axis=1)[:, np.newaxis]
    
    # Calculate Portfolio Returns
    # Matrix multiplication: weights * mean_returns
    port_returns = np.sum(all_weights * mean_returns.values, axis=1)
    
    # Calculate Portfolio Risk (Volatility)
    # Formula: sqrt(w.T * cov * w)
    port_risks = []
    for i in range(num_simulations):
        w = all_weights[i]
        # Dot product: w.T @ cov_matrix @ w
        variance = np.dot(w.T, np.dot(cov_matrix, w))
        port_risks.append(np.sqrt(variance))
    
    port_risks = np.array(port_risks)
    
    # Calculate Sharpe Ratio
    sharpe_ratios = (port_returns - risk_free_rate) / port_risks
    
    # Prepare results DataFrame
    sim_data = {
        'Return': port_returns,
        'Risk': port_risks,
        'Sharpe Ratio': sharpe_ratios
    }
    
    # Add weights to the results
    for idx, ticker in enumerate(log_returns.columns):
        sim_data[ticker] = all_weights[:, idx]
        
    results_df = pd.DataFrame(sim_data)
    
    # Identify the portfolio with the max Sharpe Ratio
    max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
    optimal_portfolio = results_df.loc[max_sharpe_idx]
    
    return results_df, optimal_portfolio