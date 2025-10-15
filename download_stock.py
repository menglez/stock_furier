import datetime
import yfinance as yf
import pandas as pd


end_date = datetime.date.today()
start_date = datetime.datetime(2000, 1, 1)


def download(etf='SPY', start_date=start_date, end_date=end_date):

    try:

        df_data = yf.download([etf], start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        df_data = df_data.reset_index()

        full_dates = pd.date_range(start=df_data['Date'].min(), end=df_data['Date'].max(), freq='D')
        df_data = df_data.set_index('Date')
        data_full = df_data.reindex(full_dates)

        # Interpolate missing values (linear by default)
        #data_full = data_full.interpolate(method='linear')

        # Forward fill missing values (e.g., weekends and holidays)
        data_full = data_full.ffill()

        data_full.index.name = 'Date'

        return data_full

    except Exception as e:

        print(f"An error occurred: {e}")
        return None