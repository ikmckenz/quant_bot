import pandas as pd
import datetime as dt
import sqlalchemy as sq
import requests, pytz, configparser

# Configuration section
config = configparser.ConfigParser()
config.read('config.txt')

db_update_freq = 3  # Days


def connect_db():
    # A simple connection that I use everywhere
    url = 'postgresql://mybot:' + config['keys']['postgres-pass'] + '@localhost:5432/quantbot'
    engine = sq.create_engine(url)
    meta = sq.MetaData(bind=engine, reflect=True)
    return engine, meta


def clean_av_prices(x: pd.DataFrame) -> pd.DataFrame:
    # Clean the data from AlphaVantage
    x.rename(columns= lambda name: str(name)[3:], inplace=True)
    x['date'] = pd.to_datetime(x.index, format='%Y-%m-%d')
    x.reset_index(drop=True, inplace=True)
    x.rename(columns={'dividend amount': 'div',
                      'adjusted close': 'adj_close',
                      'split coefficient': 'split'},
             inplace=True)
    x = x[:-1]
    return x


def db_last_update() -> pd.DataFrame:
    # Returns a dataframe with tickers and when the last info was
    engine, meta = connect_db()
    sql = "select distinct on (\"ticker\") ticker, date from prices order by ticker, date desc nulls last, date;"
    df = pd.read_sql(sql, engine)
    return df


def db_drop_ticker(ticker: str):
    # Drop the selected ticker from the database
    engine, meta = connect_db()
    sql = 'delete from prices where ticker = \'%s\';' % ticker
    engine.execute(sql)


def get_iex_stats(ticker: str) -> dict:
    # Hit the IEX API to get information about a ticker
    url = 'https://api.iextrading.com/1.0/stock/' + ticker + '/price'
    try:
        price = float(requests.get(url).text)
    except ValueError:
        db_drop_ticker(ticker)
        return {}
    url = 'https://api.iextrading.com/1.0/stock/' + ticker + '/stats'
    data = requests.get(url).json()
    nydt = pytz.timezone('UTC').localize(dt.datetime.utcnow())
    nydt = nydt.astimezone(pytz.timezone('America/New_York'))

    entry = {
        'ticker': data['symbol'],
        'name': data['companyName'],
        'beta': data['beta'],
        'price': price,
        'mktcap': int(data['marketcap']),
        'updated': nydt
    }
    return entry


def update_ticker_db(ticker: str):
    # Update db with IEX entry
    ticker = ticker.upper()
    # Connect to db
    engine, meta = connect_db()
    ticker_table = meta.tables['tickers']
    conn = engine.connect()
    s = ticker_table.select().where(ticker_table.c.ticker == ticker)
    s = sq.sql.expression.exists(s).select()
    exists = conn.execute(s).scalar()
    if exists:
        entry = get_iex_stats(ticker)
        if len(entry) == 0:
            conn.close()
            return 1
        s = sq.sql.expression.update(ticker_table). \
            where(ticker_table.c.ticker == entry['ticker']). \
            values(entry)
        conn.execute(s)
    else:
        entry = get_iex_stats(ticker)
        if len(entry) == 0:
            conn.close()
            return 1
        s = sq.sql.expression.insert(ticker_table).values(entry)
        conn.execute(s)
    conn.close()
    return 0


def import_full_history(ticker: str) -> int:
    status = update_ticker_db(ticker)
    if status == 1:
        return status
    # Pass in a ticker to grab full history from AV, put in the prices table
    engine, meta = connect_db()
    conn = engine.connect()
    # Check existence of ticker
    prices = meta.tables['prices']
    s = prices.select().where(prices.c.ticker == ticker)
    s = sq.sql.expression.exists(s).select()
    ret = conn.execute(s).scalar()
    if ret:
        conn.close()
        status = 0
    else:
        print('Getting full history of %s from AlphaVantage' % ticker)
        query_string = ('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED'
                        '&symbol=%s&outputsize=full&apikey=%s' % (ticker, config['keys']['alphavantage']))
        response = requests.get(query_string)
        data = response.json()
        if 'Meta Data' and 'Time Series (Daily)' in data:
            symbol = data['Meta Data']['2. Symbol']
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df = clean_av_prices(df)
            df['ticker'] = symbol
            df.to_sql('prices', engine, index=False, if_exists='append')
            status = 0
        elif 'Error Message' in data:
            status = 1
        conn.close()
    return status


def update_price_data() -> None:
    # Update all the data in the prices table
    engine, meta = connect_db()
    df = db_last_update()
    todaysdate = dt.datetime.today().date()
    for index, row in df.iterrows():
        needed_data = (todaysdate - row['date']).days
        if needed_data >= 100:  # AV compact download returns only 100 data points
            print('Updating %s' % row['ticker'])
            print('Too much data missing. Purging and re-downloading.')
            db_drop_ticker(row['ticker'])
            import_full_history(row['ticker'])
        else:
            if todaysdate > row['date'] + dt.timedelta(days=db_update_freq):
                print('Updating %s' % row['ticker'])
                query_string = ('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED'
                                '&symbol=%s&apikey=%s' % (row['ticker'], config['keys']['alphavantage']))
                response = requests.get(query_string)
                data = response.json()
                if 'Meta Data' and 'Time Series (Daily)' in data:
                    symbol = data['Meta Data']['2. Symbol']
                    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                    df = clean_av_prices(df)
                    df['ticker'] = symbol
                    df = df[df['date'] > row['date']]
                    print('Adding %d rows to db for ticker %s' % (len(df), row['ticker']))
                    df.to_sql('prices', engine, index=False, if_exists='append')
                elif 'Error Message' in data:
                    print('Woah, bad error. Cant download existing ticker %s' % row['ticker'])


if __name__ == '__main__':
    # We can run this file as a script to update the price data
    update_price_data()
