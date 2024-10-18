import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import norm
import numpy as np

class OptionChainRetriever:
    def __init__(self, r=0.01):
        """
        Initialize the OptionChainRetriever with the given risk-free rate.
        
        Parameters:
        r : float : Risk-free interest rate (default is 0.01).
        """
        self.r = r

    


    # Function to retrieve S&P 500 tickers
    def get_sp500_tickers(self):
        """
        Get a list of S&P 500 tickers from Wikipedia.
        """
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_table = pd.read_html(url)
        sp500_df = sp500_table[0]
        return sp500_df['Symbol'].tolist()

    # Function to get QQQ tickers
    def get_qqq_tickers(self):
        """
        Get a list of tickers for QQQ.
        """
        qqq_ticker = 'QQQ'
        qqq_holdings = yf.Ticker(qqq_ticker).history(period='1d')
        if qqq_holdings.empty:
            raise ValueError(f"No data found for {qqq_ticker}.")
        return [qqq_ticker]

    def get_stock_price(self, ticker):
        """
        Fetches the current stock price for a given ticker using yfinance.
        
        Parameters:
        ticker : str : The stock ticker symbol.
        
        Returns:
        float : The current stock price.
        """
        ticker_data = yf.Ticker(ticker).history(period='1d')
        if ticker_data.empty:
            raise ValueError(f"No historical data found for {ticker}.")
        return ticker_data['Close'].iloc[0]

    def black_scholes_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate the Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho) for call or put options.
        
        Parameters:
        S : float : Current stock price.
        K : float : Option strike price.
        T : float : Time to expiration in years.
        r : float : Risk-free interest rate.
        sigma : float : Implied volatility.
        option_type : str : 'call' or 'put' (default is 'call').
        
        Returns:
        tuple : Delta, Gamma, Theta, Vega, Rho.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        elif option_type == 'put':
            delta = norm.cdf(d1) - 1
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return delta, gamma, theta, vega, rho

    def retrieve_option_chain(self, tickers, expiration_dates, iv_threshold=None, theta_price_threshold=None, option_type=None, strike_price_range=None):
        """
        Retrieve and filter options based on Implied Volatility (IV), absolute Theta/Price, and optionally a strike price range.
        
        Parameters:
        tickers : list : List of stock tickers.
        expiration_dates : list : List of expiration dates to check.
        iv_threshold : float : Minimum acceptable IV for filtering (high IV). Default is None (no filtering).
        theta_price_threshold : float : Minimum acceptable Theta/Price for filtering (high absolute Theta/Price). Default is None (no filtering).
        option_type : str : 'call', 'put', or None (for both). Default is None (both types).
        strike_price_range : float : Percentage range for strike price filtering (e.g., 20 for 20%). Default is None (no filtering).
        
        Returns:
        result_df : DataFrame : Filtered options based on the given criteria.
        """
        result_df = pd.DataFrame()

        for ticker in tickers:
            try:
                current_price = self.get_stock_price(ticker)

                for expiration_date in expiration_dates:
                    stock = yf.Ticker(ticker)
                    options = stock.option_chain(expiration_date)

                    # Filter based on user choice for calls, puts, or both
                    option_types = []
                    if option_type == 'call':
                        option_types.append(options.calls)
                    elif option_type == 'put':
                        option_types.append(options.puts)
                    else:
                        option_types = [options.calls, options.puts]
                    
                    filtered_options = []
                    for option_data in option_types:
                        for _, row in option_data.iterrows():
                            strike_price = row['strike']
                            last_price = row['lastPrice']
                            implied_volatility = row['impliedVolatility']
                            T = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days / 365  # Time to expiration in years

                            # Calculate Greeks using Black-Scholes model
                            delta, gamma, theta, vega, rho = self.black_scholes_greeks(current_price, strike_price, T, self.r, implied_volatility, 'call' if 'call' in row['contractSymbol'].lower() else 'put')

                            # Use absolute value of Theta for comparison
                            abs_theta = abs(theta)

                            # Calculate Theta/Price using the absolute value of Theta
                            theta_price_ratio = abs_theta / last_price if last_price > 0 else None

                            # Skip IV and Theta/Price filtering if no threshold is provided
                            if (iv_threshold is not None and implied_volatility < iv_threshold) or \
                               (theta_price_threshold is not None and theta_price_ratio is not None and theta_price_ratio < theta_price_threshold):
                                continue  # Skip this option if IV or Theta/Price is below the threshold

                            # Strike price filtering: Apply range if specified by the user
                            if strike_price_range is not None:
                                lower_bound = current_price * (1 - strike_price_range / 100)
                                upper_bound = current_price * (1 + strike_price_range / 100)
                                if not (lower_bound <= strike_price <= upper_bound):
                                    continue  # Skip this option if the strike price is outside the range

                            # If it passes the filters, add the option to the results
                            filtered_options.append({
                                'Ticker': ticker.upper(),
                                'Type': 'Call' if 'call' in row['contractSymbol'].lower() else 'Put',
                                'Strike': strike_price,
                                'Last Price': last_price,
                                'Implied Volatility': implied_volatility,
                                'Theta/Price': theta_price_ratio,
                                'Delta': delta,
                                'Gamma': gamma,
                                'Theta': theta,
                                'Vega': vega,
                                'Rho': rho
                            })

                    # Append filtered results to result dataframe
                    if len(filtered_options) > 0:
                        df = pd.DataFrame(filtered_options)
                        df['Expiration Date'] = expiration_date
                        result_df = pd.concat([result_df, df], ignore_index=True)
            
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        return result_df
