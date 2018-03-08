import numpy as np
import pandas as pd
from database_updates import connect_db


def pandas2post(df):
    # Turn a pandas dataframe into a formatted reddit post
    message = 'Index | '
    below_columns = '-|'
    for col in list(df):
        message = message + col + ' | '
        below_columns = below_columns + '-|'
    message = message[:-3]
    below_columns = below_columns[:-1]
    message = message + '\n'
    message = message + below_columns + '\n'
    for index, row in df.iterrows():
        message = message + str(index) + ' | '
        for i in row.tolist():
            message = message + str(i) + ' | '
        message = message[:-3]
        message = message + '\n'
    return message


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
    sql = 'select distinct date from prices order by date desc limit 252;'
    df = pd.read_sql(sql, engine)
    df = df.iloc[::-1]
    df.set_index('date', drop=True, inplace=True)
    for ticker in tickers:
        sql = 'select date, adj_close from prices where ticker = \'%s\' order by date desc limit 252;' % ticker
        data = pd.read_sql(sql, engine)
        data = data.iloc[::-1]
        data.set_index('date', drop=True, inplace=True)
        data.rename(columns={'adj_close': ticker}, inplace=True)
        df = df.join(data, how='inner')
    return df


def get_multi_ticker_adj_volume(tickers):
    engine, meta = connect_db()
    sql = 'select distinct date from prices order by date desc limit 252;'
    df = pd.read_sql(sql, engine)
    df = df.iloc[::-1]
    df.set_index('date', drop=True, inplace=True)
    for ticker in tickers:
        sql = 'select date, volume from prices where ticker = \'%s\' order by date desc limit 252;' % ticker
        data = pd.read_sql(sql, engine)
        data = data.iloc[::-1]
        data.set_index('date', drop=True, inplace=True)
        data.rename(columns={'volume': ticker}, inplace=True)
        df = df.join(data, how='inner')
    return df


def simple_vol(ticker):
    df = get_single_ticker_year_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    df['return'] = df['close'].pct_change()
    vol = np.sqrt(252)*df['return'].std(skipna=True) * 100
    return 'Volatility of %s from %s to %s was %.5f%%' % (ticker, first_date, last_date, vol)


def garman_klass_vol(ticker):
    df = get_single_ticker_year_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    vol = 0
    for index, row in df.iterrows():
        vol += (0.5*np.log(row['high']/row['low'])**2) - \
               ((2*np.log(2) - 1)*np.log(row['close']/row['open'])**2)
    vol = np.sqrt(vol)
    vol *= 100
    return 'Garman-Klass volatility of %s from %s to %s was %.5f%%' % (ticker, first_date, last_date, vol)


def get_avg_volume(ticker):
    df = get_single_ticker_year_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    avg_volume = df['volume'].mean()
    avg_volume = "{:,.2f}".format(avg_volume)
    return 'Average daily volume of %s from %s to %s was %s' % (ticker, first_date, last_date, avg_volume)


def price_correlation_matrix(tickers):
    df = get_multi_ticker_adj_close(tickers)
    df = df.corr()
    message = 'Price correlation matrix:\n\n'
    message = message + pandas2post(df)
    return message


def volume_correlation_matrix(tickers):
    df = get_multi_ticker_adj_volume(tickers)
    df = df.corr()
    message = 'Volume correlation matrix:\n\n'
    message = message + pandas2post(df)
    return message
