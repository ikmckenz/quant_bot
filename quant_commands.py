# TODO: Generate multiple ticker returns sql query
# TODO: Use that to create correlation matrix.

import numpy as np
import pandas as pd
from database_updates import connect_db


def get_single_ticker_year_data(ticker):
    # First, let's grab the last 252 data points
    engine, meta = connect_db()
    sql = 'select * from prices where ticker = \'%s\' order by date desc limit 252;' % ticker
    df = pd.read_sql(sql, engine)
    # Flip it the right way around
    df = df.iloc[::-1]
    df.reset_index(drop=True, inplace=True)
    return df


def get_multi_ticker_adj_close(tickers):
    engine, meta = connect_db()
    sql = ''
    # Ummmmm


def simple_vol(ticker):
    df = get_single_ticker_year_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    df['return'] = df['adj_close'].pct_change()
    vol = np.sqrt(252)*df['return'].std(skipna=True) * 100
    return 'Volatility of %s from %s to %s was %.5f%%' % (ticker, first_date, last_date, vol)


def garman_klass_vol(ticker):
    df = get_single_ticker_year_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    vol = 0
    for index, row in df.iterrows():
        vol += (0.5*np.log(row['adj_high']/row['adj_low'])**2) - \
               ((2*np.log(2) - 1)*np.log(row['adj_close']/row['adj_open'])**2)
    vol = np.sqrt(vol)
    vol *= 100
    return 'Garman-Klass volatility of %s from %s to %s was %.5f%%' % (ticker, first_date, last_date, vol)


def get_avg_volume(ticker):
    df = get_single_ticker_year_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    avg_volume = df['adj_volume'].mean()
    avg_volume = "{:,.2f}".format(avg_volume)
    return 'Average daily volume from %s to %s was %s' % (first_date, last_date, avg_volume)
