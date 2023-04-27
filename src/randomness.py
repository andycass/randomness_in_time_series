import pandas as pd
import matplotlib.pyplot as plt

from numpy import cumsum, log, polyfit, sqrt, std, subtract, min, mean, zeros, ptp
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import runs
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kstest
from arch.unitroot import bds

class Randomness:
    def __init__(self, df: pd.DataFrame):
        self.df = self.add_returns_to_df(df)

    def add_returns_to_df(self):
        self.df['Returns'] = self.df['Close'] / self.df['Close'].shift(1) - 1
        return self.df.dropna(inplace=True)
    
    def binary_returns(self):
        returns =  self.df['Returns'].values.astype(float)
        returns_binary = [ 1 if x >= 0 else 0 for x in returns]
        return returns_binary

    def runs_test_one(self):
        '''
        takes the 
        '''
        returns_binary = self.binary_returns()
        (z_stat, p_value) = runstest_1samp(returns_binary[:10], correction=False)
        z_stat = round(z_stat, 3)
        p_value = round(p_value, 3)
        is_reject_runs = True if p_value < 0.05 else False
        print(f"Z-Statistic: {z_stat}")
        print(f"P-Value: {p_value}")
        print(f"Reject Null: {is_reject_runs}")
        print(f"Observable Runs Exceeds Excpected Runs by: {z_stat} Standard Deviations")
        print("Not Random") if is_reject_runs else print("Random")

    def bds_test(self):
        '''

        '''
        returns = self.df['Returns']
        bds_test = bds(returns[-500:], distance=2)
        bds_stat = float(bds_test[0])
        pvalue = float(bds_test[1])
        print("BDS Test Statistic: ", round(bds_stat, 3))
        print("BDS P-Value: ", round(pvalue, 3))
        print("Not Random") if pvalue < 0.05 else print("Random")

    def hurst_test(self, min_lag=1, max_lag=100):
        '''
        "Whether a market tends to trend, mean revert, or is just random is valuable information for a trader. 
        While the Hurst exponent isn't an entry signal in and of itself, 
        it can serve as a filter on top of a system. 
        Given that market regimes can shift over time to favor one approach or the other,
        overlaying your model with a Hurst filter could help prevent your algorithm from buying a breakout in a mean reverting market 
        or shorting ahead of a pullback when the market is moving to new highs."
        Find Your Best Market to Trade With the Hurst Exponent (referenced below)
        If Hurst = 0.5, then the market is random.
        If Hurst > 0.5, then there is evidence of a trending market.
        If Hurst < 0.5, then there is evidence of a mean reverting market.
        '''
        prices = self.df["Close"].values
        lags = range(min_lag, max_lag)
        tau = [sqrt(std(subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        poly = polyfit(log(lags), log(tau), 1)
        return poly[0]*2.0
    
    def ad_fuller_test(self):
        '''
        '''
        returns = self.df['Returns']
        dftest = adfuller(returns)
        p_value = dftest[1]
        t_test = dftest[0] < dftest[4]["1%"]
        print(p_value, t_test)
        print("If < 0.05 and True then we can reject the null hypothesis and conclude that the index is stationary")
        
    #####################

    def adfuller_test(series):
        adf_result = adfuller(series)
        p_value = adf_result[1]
        return p_value

    def runs_test_two(self):
        rtest = runs.Runs(self.series)
        p_value = rtest.pvalue
        return p_value
    
    def autocorrelation_test(series):
        '''
        Autocorrelation Test:
        '''
        dwtest = durbin_watson(series)
        return dwtest 
    
    def ljung_box_test(series, lags=10):
        lbtest = acorr_ljungbox(series, lags=lags)
        p_value = lbtest[1][-1]
        return p_value
    
    def kolmogorov_smirnov_test(series):
        kstest_result = kstest(series, 'uniform')
        p_value = kstest_result.pvalue
        return p_value
    
    def hurst_exponent(self, series, lag_range=None):
        """
        Estimate the Hurst exponent of a time series using the rescaled range analysis.
        
        Args:
            series (np.ndarray or pd.Series): The time series data.
            lag_range (tuple or list): A tuple or list of two integers specifying the minimum and maximum lags to use for the R/S analysis.
        
        Returns:
            The estimated Hurst exponent.
        """
        # Convert the series to a numpy array if needed
        if isinstance(series, pd.Series):
            series = series.values
            
        # Compute the cumulative deviation of the series
        deviations = cumsum(series - mean(series))
        
        # Initialize variables for the R/S analysis
        if lag_range is None:
            lag_range = [2, int(len(series) / 2)]
        min_lag, max_lag = lag_range
        num_lags = max_lag - min_lag + 1
        rs_values = zeros(num_lags)
        lag_values = zeros(num_lags)
        
        # Perform the R/S analysis for each lag
        for i, lag in enumerate(range(min_lag, max_lag+1)):
            window_size = int(len(series) / lag)
            reshaped = deviations[:window_size*lag].reshape(window_size, lag)
            window_range = ptp(reshaped, axis=1)
            window_std = std(reshaped, axis=1, ddof=1)
            rs_values[i] = mean(window_range / window_std)
            lag_values[i] = lag
            
        # Fit a line to the R/S values in log-log space
        log_rs = log(rs_values)
        log_lag = log(lag_values)
        slope, _, _, _, _ = polyfit(log_lag, log_rs, 1, full=True)
        
        # Return the Hurst exponent (slope + 1)
        return slope + 1
    
    def bds_test(series, max_lag=None, lag_list=None):
        """
        Conduct the BDS test for independence in a time series.
        
        Args:
            series (np.ndarray or pd.Series): The time series data.
            max_lag (int): The maximum number of lags to test (default is None, which uses the square root of the series length).
            lag_list (list): A list of lags to test (default is None, which uses a range of 1 to max_lag).
        
        Returns:
            The p-value of the BDS test.
        """
        # Convert the series to a numpy array if needed
        if isinstance(series, pd.Series):
            series = series.values
            
        # Compute the default values for max_lag and lag_list if needed
        if max_lag is None:
            max_lag = int(sqrt(len(series)))
        if lag_list is None:
            lag_list = list(range(1, max_lag+1))
        
        # Conduct the BDS test for each lag
        results = bds(series, lags=lag_list)
        p_values = results.pvalues
        
        # Return the smallest p-value
        return min(p_values)
    

    ### PLOT ###

    # Plot the ACF (Auto Correction function)
    def plot_auto_correction(series):
        plot_acf(lags=20)

    # Plot the PACF (Partial Auto Correction function)
    def plot_partial_aut_correction(series):
        plot_pacf(series, lags=20)
    
    def plot_adf(series):
        # Conduct the ADF test
        result = adfuller(series)
        adf_stat = result[0]
        p_value = result[1]

        # Plot the time series with ADF test results
        plt.plot(series)
        plt.title(f'ADF Test: p-value = {p_value:.4f}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()
