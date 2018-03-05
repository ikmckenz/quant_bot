# TODO: add ticker queue to update after close (file as a stack maybe)

import numpy as np
import pandas as pd
import datetime as dt
import sqlalchemy as sq
import requests, json, pytz
from multiprocessing import cpu_count, Pool
import quandl

with open('key.txt') as file:
    for line in file:
        my_quandl_key = line

quandl.ApiConfig.api_key = my_quandl_key
num_cores = cpu_count()
num_partitions = num_cores * 2
quandl_update_freq = 0 # Days


def connect_db():
    engine = sq.create_engine('postgresql://mybot:botpass@localhost:5432/postgres')
    meta = sq.MetaData(bind=engine, reflect=True)
    return engine, meta


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def to_datetime(x):
    x['date'] = pd.to_datetime(x['date'], format='%Y-%m-%d')
    return x


def clean_quandl_prices(x):
    x.rename(columns={"ex-dividend": "div"}, inplace=True)

    if x.shape[0] > 1000:
        x = parallelize_dataframe(x, to_datetime)
    else:
        x = to_datetime(x)
    x.set_index('date', drop=True, inplace=True)
    return x


def last_updated_quandl():
    # Returns a dataframe with tickers and when the last info was
    engine, meta = connect_db()
    sql = "select distinct on (\"ticker\") ticker, date from prices order by ticker, date desc nulls last, date;"
    df = pd.read_sql(sql, engine)
    df = to_datetime(df)
    return df


def get_google_fin(ticker):
    url = 'https://finance.google.com/finance?q=' + ticker + '&output=json'
    response = requests.get(url).text
    response = response[4:]
    data = json.loads(response)
    nydt = pytz.timezone('UTC').localize(dt.datetime.utcnow())
    nydt = nydt.astimezone(pytz.timezone('America/New_York'))
    nydt = nydt.strftime("%Y-%m-%d")
    entry = {
        'ticker': data[0]['symbol'],
        'exchange': data[0]['exchange'],
        'name': data[0]['name'],
        'beta': float(data[0]['beta']),
        'price': float(data[0]['l']),
        'changep': float(data[0]['cp']),
        'mktcap': data[0]['mc'],
        'updated': nydt
    }
    return entry


def update_ticker(ticker):
    # Update db with google finance entry
    ticker = ticker.upper()
    # Connect to db
    engine, meta = connect_db()
    ticker_table = meta.tables['tickers']
    conn = engine.connect()
    s = ticker_table.select().where(ticker_table.c.ticker == ticker)
    s = sq.sql.expression.exists(s).select()
    ret = conn.execute(s).scalar()

    if ret:
        last_updated = "select updated from tickers where ticker = \'" + ticker + "\'"
        (last_updated,) = conn.execute(last_updated)
        last_updated = last_updated[0]
        nydt = pytz.timezone('UTC').localize(dt.datetime.utcnow())
        nydt = nydt.astimezone(pytz.timezone('America/New_York'))
        past_four = nydt.time() > dt.time(16, 1)
        if (last_updated + dt.timedelta(days=1)) <= nydt.date():
            if past_four:
                entry = get_google_fin(ticker)
                s = sq.sql.expression.update(ticker_table). \
                    where(ticker_table.c.ticker == entry['ticker']). \
                    values(entry)
                conn.execute(s)
            else:
                entry = get_google_fin(ticker)
                s = sq.sql.expression.update(ticker_table). \
                    where(ticker_table.c.ticker == entry['ticker']). \
                    values(entry)
                conn.execute(s)
                # Update ticker later
                # update_ticker_later(ticker)
    else:
        entry = get_google_fin(ticker)
        s = sq.sql.expression.insert(ticker_table).values(entry)
        conn.execute(s)
        nydt = pytz.timezone('UTC').localize(dt.datetime.utcnow())
        nydt = nydt.astimezone(pytz.timezone('America/New_York'))
        past_four = nydt.time() > dt.time(16, 1)
        if not past_four:
            pass
            # Update ticker later
            # update_ticker_later(ticker)
    conn.close()


def import_price_history(ticker):
    # Pass in a ticker to grab full history from Quandl, put in the prices table
    # Will throw error if ticker exists
    engine, meta = connect_db()
    conn = engine.connect()
    # Check existence of ticker
    prices = meta.tables['prices']
    s = prices.select().where(prices.c.ticker == ticker)
    s = sq.sql.expression.exists(s).select()
    ret = conn.execute(s).scalar()
    if ret:
        print('Error: Ticker exists.')
        conn.close()
        # Throw error? Return 1?
        return 1
    else:
        df = quandl.get_table('WIKI/PRICES', ticker=ticker)
        df = clean_quandl_prices(df)
        df.to_sql('prices', conn, if_exists='append')
        conn.close()
        return 0


def update_price_data():
    # Update all the data in the prices table
    engine, meta = connect_db()
    df = last_updated_quandl()
    df['date'] = df['date'] + dt.timedelta(days=1)
    last_updated_dt = df['date']
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    triggerdate = dt.datetime.today() + dt.timedelta(days=quandl_update_freq)
    for index, row in df.iterrows():
        if triggerdate > last_updated_dt[index]:
            data = quandl.get_table('WIKI/PRICES',
                                    ticker=row['ticker'],
                                    date={'gte': row['date']})
            if len(data) == 0:
                # Probably switch to logging instead of printing
                print('Nothing to update for %s' % row['ticker'])
            else:
                data = clean_quandl_prices(data)
                print('Updating %s' % row['ticker'])
                data.to_sql('prices', engine, if_exists='append')
