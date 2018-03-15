# Quant Bot

This is a toy project I used to learn a little bit about SQL. 
Basically it's a Reddit bot that will return quantitative analytics about a stock.
I've got three main tables in my Postgres database: prices, tickers, and posts.
Prices keeps all my historic price data, tickers keeps metadata about each stock, and posts is a simple list of Reddit posts I've already responded to.
Someone could use this code to start their own bot, or as their own personal quantitative analyst. 

## Reddit Usage

#### Volatility
See the [volatility](https://www.investopedia.com/terms/v/volatility.asp) of a stock over the last year.
Use `!quant_bot vol TICKER`
where TICKER is the stock you are interested in.

#### Histogram
Get a histogram of 5 years of weekly returns with some statistical data.
Use `!quant_bot hist TICKER`
where TICKER is the stock you are interested in.

#### Peer Analysis
Get a detailed [peer](https://www.investopedia.com/terms/p/peer-group.asp) analysis, including a graph of normalized returns for all peers, a table of peers with analytical data, and a correlation matrix generated with one year of price data.
Use `!quant_bot peers TICKER`
where TICKER is the stock you are interested in.

#### More
I've already implemented a couple more things in the code but haven't yet 'activated' them. 
Stay tuned for updates on this page.

## Code Usage

Because this was a learning experience, the SQL code is spaghetti.
I use various different ways to interact with my database including the SQLAlchemy ORM, executing raw SQL queries with SQLAlchemy, and executing SQL queries using Pandas.
In addition to this the non-SQL code is spaghetti because I was trying to learn SQL and hacked the other stuff together. 
Overall the code is a little bit messy.

With this warning in mind, if you'd still like to use the code yourself you have to set a few things up first.
* The database. You'll need PostgreSQL and a database with three tables: Prices, Tickers, and Posts. If you don't want to set this up manually, you can use the file `quantbot.dump` to set up the database.
* You will need to put your various API keys into a file called config.txt with the following layout:
```
[keys]
postgres-pass: XXXXXXX
alphavantage: XXXXXXX
imgur-client-id: XXXXXXX
imgur-client-secret: XXXXXXX
reddit-client-id: XXXXXXX
reddit-secret: XXXXXXX
reddit-password: XXXXXXX
reddit-username: XXXXXXX
``` 
The order of the entries doesn't matter and you only need what you're going to use.
For example, if you are going to be uploading all the images to a local server (or just viewing/saving them with your local machine), then you don't need an Imgur API key. If you aren't going to be posting on Reddit then you don't need anything from Reddit, etc.
