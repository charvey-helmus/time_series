import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot, lag_plot

def explore_time_series(df, datetime_col, target_col, resample_freq='D'):
    # Parse datetime and set index
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    df.sort_index(inplace=True)
    
    print("\n--- Dataset Info ---")
    print(df[[target_col]].info())
    print("\n--- Missing Values ---")
    print(df[[target_col]].isna().sum())
    print("\n--- Summary Statistics ---")
    print(df[[target_col]].describe())

    # Plot original
    df[[target_col]].plot(figsize=(14, 6), title="Original Time Series")
    plt.grid()
    plt.show()

    # Resample
    df_resampled = df[[target_col]].resample(resample_freq).mean()
    df_resampled.plot(figsize=(14, 6), title=f"Resampled ({resample_freq}) Time Series")
    plt.grid()
    plt.show()

    # Rolling stats
    rolling_mean = df_resampled.rolling(window=30).mean()
    rolling_std = df_resampled.rolling(window=30).std()

    plt.figure(figsize=(14,6))
    plt.plot(df_resampled, label='Resampled')
    plt.plot(rolling_mean, label='Rolling Mean (30)', color='orange')
    plt.plot(rolling_std, label='Rolling Std (30)', color='green')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.grid()
    plt.show()

    # Decomposition
    print("\n--- Decomposition ---")
    decomposition = seasonal_decompose(df_resampled.dropna(), model='additive', period=None)
    fig = decomposition.plot()
    fig.set_size_inches(14, 8)
    plt.show()

    # ADF Test
    print("\n--- Augmented Dickey-Fuller Test ---")
    result = adfuller(df_resampled.dropna()[target_col])
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")

    # Autocorrelation plot
    plt.figure(figsize=(10, 4))
    autocorrelation_plot(df_resampled.dropna()[target_col])
    plt.title("Autocorrelation Plot")
    plt.grid()
    plt.show()

    # Lag plot
    plt.figure(figsize=(6, 6))
    lag_plot(df_resampled.dropna()[target_col], lag=1)
    plt.title("Lag Plot (lag=1)")
    plt.grid()
    plt.show()
