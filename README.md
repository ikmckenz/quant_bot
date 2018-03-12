# Quant Bot

This is a toy project I used to learn a little bit about SQL. 
Basically it's a Reddit bot that will return quantitative analytics about a stock.
I've got three main tables in my Postgres database: prices, tickers, and posts.
Prices keeps all my historic price data, tickers keeps metadata about each stock, and posts is a simple list of Reddit posts I've already responded to.
Someone could use this code to start their own bot, or as their own personal quantitative analyst. 

Because this was a learning experience, the SQL code is spaghetti.
I use various different ways to interact with my database including the SQLAlchemy ORM, executing raw SQL queries with SQLAlchemy, and executing SQL queries using Pandas.
In addition to this the non-SQL code is spaghetti because I was trying to learn SQL and hacked the other stuff together. 
Overall the code is a little bit messy.
