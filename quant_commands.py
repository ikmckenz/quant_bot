# TODO: Add price performance comparison, Add CAPM comparisons

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import scipy.stats as ss
import configparser
import requests
import io
import base64
from typing import List
from database_updates import connect_db, import_full_history


# Configuration section
sns.set()
config = configparser.ConfigParser()
config.read('config.txt')


def pandas2post(df: pd.DataFrame, keep_index=1) -> str:
    # Turn a pandas dataframe into a formatted reddit post
    if keep_index:
        message = 'Index | '
    else:
        message = ''
    below_columns = '-|'
    for col in list(df):
        message = message + col + ' | '
        below_columns = below_columns + '-|'
    message = message[:-3]
    below_columns = below_columns[:-1]
    message = message + '\n'
    message = message + below_columns + '\n'
    for index, row in df.iterrows():
        if keep_index:
            message = message + str(index) + ' | '
        for i in row.tolist():
            message = message + str(i) + ' | '
        message = message[:-3]
        message = message + '\n'
    return message


def post_imgur(fig):
    pic_bytes = io.BytesIO()
    fig.savefig(pic_bytes, format='png', bbox_inches='tight')
    pic_bytes.seek(0)
    pic_string = base64.b64encode(pic_bytes.getvalue())
    url = 'https://api.imgur.com/3/upload.json'
    imgur_id = "Client-ID %s" % config['keys']['imgur-client-id']
    headers = {"Authorization": imgur_id}
    resp = requests.post(url,
                         headers=headers,
                         data={
                             'image': pic_string,
                             'type': 'base64'
                         })
    resp = resp.json()
    return resp['data']['link']


def get_single_ticker_data(ticker: str, years=1):
    status = import_full_history(ticker)
    # First, let's grab the last 252 data points
    engine, meta = connect_db()
    sql = 'select * from prices where ticker = \'%s\' order by date desc limit %d;' % (ticker, 252*years)
    df = pd.read_sql(sql, engine)
    # Flip it the right way around
    df = df.iloc[::-1]
    df.reset_index(drop=True, inplace=True)
    return df


def get_multi_ticker_adj_close(tickers: List[str], years=1):
    for ticker in tickers:
        import_full_history(ticker)
    engine, meta = connect_db()
    sql = 'select distinct date from prices order by date desc limit %d;' % 252*years
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


def get_multi_ticker_adj_volume(tickers: List[str]):
    for ticker in tickers:
        import_full_history(ticker)
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


def simple_vol(ticker: str):
    df = get_single_ticker_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    df['return'] = df['close'].pct_change()
    vol = np.sqrt(252)*df['return'].std(skipna=True) * 100
    return 'Volatility of %s from %s to %s was %.5f%%' % (ticker, first_date, last_date, vol)


def garman_klass_vol(ticker: str):
    df = get_single_ticker_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    vol = 0
    for index, row in df.iterrows():
        vol += (0.5*np.log(row['high']/row['low'])**2) - \
               ((2*np.log(2) - 1)*np.log(row['close']/row['open'])**2)
    vol = np.sqrt(vol)
    vol *= 100
    return 'Garman-Klass volatility of %s from %s to %s was %.5f%%' % (ticker, first_date, last_date, vol)


def get_avg_volume(ticker: str):
    df = get_single_ticker_data(ticker)
    last_date = max(df['date'])
    first_date = min(df['date'])
    avg_volume = df['volume'].mean()
    avg_volume = "{:,.2f}".format(avg_volume)
    return 'Average daily volume of %s from %s to %s was %s' % (ticker, first_date, last_date, avg_volume)


def price_correlation_matrix(tickers: List[str]):
    df = get_multi_ticker_adj_close(tickers)
    df = df.corr()
    df = df.round(decimals=4)
    message = 'Price correlation matrix:\n\n'
    message = message + pandas2post(df)
    return message


def volume_correlation_matrix(tickers: List[str]):
    df = get_multi_ticker_adj_volume(tickers)
    df = df.corr()
    message = 'Volume correlation matrix:\n\n'
    message = message + pandas2post(df)
    return message


def ticker_histogram(ticker: str):
    df = get_single_ticker_data(ticker, 5)
    last_date = max(df['date'])
    df = df[['date', 'close']]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', drop=True, inplace=True)
    df = df.resample('W-MON').last()
    df['return'] = df['close'].pct_change()
    df.dropna(inplace=True)
    # Analytics
    mean = df['return'].mean()
    std = df['return'].std()
    skew = df['return'].skew()
    kurtosis = df['return'].kurtosis()
    # Plot
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(df['return'], bins=50)
    ticks = ax.get_xticks()
    ax.set_xticklabels(['{:.2f}%'.format(x*100) for x in ticks])
    ax.set_xlabel('Returns')
    ax.xaxis.set_tick_params(size=3)
    ax.set_ylabel('Count')
    ax.set_title('%s Weekly Returns Over 5 Years' % ticker)
    # Fit normal
    y = mlab.normpdf(bins, mean, std)
    ax.plot(bins, y, 'r--')
    plot_text = "Updated: %s\nMean: %.3f\nStd: %.3f\nSkew: %.3f\nKurtosis: %.3f" \
                % (last_date, mean, std, skew, kurtosis)
    ax.annotate(plot_text, xy=(0, 1), xytext=(+15, -15), fontsize=10,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='left', verticalalignment='top')
    # Post to Imgur
    link = post_imgur(fig)
    plt.clf()
    message = '[%s 5 years of weekly returns as of %s](%s)' % (ticker, last_date, link)
    return message


def normalized_returns(tickers: List[str]):
    df = get_multi_ticker_adj_close(tickers)
    df.index = pd.to_datetime(df.index)
    df = df.pct_change()
    df.fillna(value=0, inplace=True)
    df = df + 1
    df = df.cumprod()
    # Plot
    fig, ax = plt.subplots()
    for i in list(df):
        ax.plot(df.index, df[i], label=i)
    # Other formatting
    fig.autofmt_xdate()
    ax.set_title('Normalized Returns')
    # Add our legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Post to Imgur
    link = post_imgur(fig)
    plt.clf()
    message = '[One year of normalized returns](%s)' % link
    return message


def info_list(tickers: List[str]) -> pd.DataFrame:
    engine, meta = connect_db()
    sql = 'select * from tickers where '
    for ticker in tickers:
        sql = sql + 'ticker = \'%s\' or ' % ticker
    sql = sql[:-4] + ';'
    df = pd.read_sql(sql, engine)
    message = pandas2post(df, keep_index=0)
    return message


def peer_comp(ticker: str):
    url = 'https://api.iextrading.com/1.0/stock/' + ticker + '/peers'
    resp = requests.get(url)
    if resp.status_code == 200:
        peers = resp.json()
        peers.append('SPY')
    else:
        peers = ['SPY']
    peers.insert(0, ticker)
    message = 'Peer comparison for %s:\n\n' % ticker
    norm_ret = normalized_returns(peers)
    message = message + norm_ret + '\n\n'
    message = message + 'Peers: \n\n'
    peer_table = info_list(peers)
    message = message + peer_table + '\n\n'
    message = message + price_correlation_matrix(peers)
    return message
