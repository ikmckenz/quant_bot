import pandas as pd
import praw
import re
import configparser
from database_updates import connect_db, import_full_history
from quant_commands import ticker_histogram, peer_comp, simple_vol


# Configuration section
config = configparser.ConfigParser()
config.read('config.txt')

# Connect to Reddit
reddit = praw.Reddit(client_id=config['keys']['reddit-client-id'],
                     client_secret=config['keys']['reddit-secret'],
                     password=config['keys']['reddit-password'],
                     user_agent='quant_bot by /u/ikmckenz',
                     username='quant_bot')

# Get comments already replied to
engine, meta = connect_db()
sql = "select distinct orig as comments from posts;"
replied = pd.read_sql(sql, engine)


def good_ticker(ticker: str) -> int:
    # Very basic SQL Injection prevention. Like very basic.
    print("Getting ticker %s" % ticker)
    if re.search("DROP", ticker):
        return 0
    if ticker.isalpha() and (len(ticker) < 6):
        status = import_full_history(ticker)
        if status == 0:
            return 1  # If no error, return 1. Very confusing, but makes the if-statements nicer below.
        else:
            return 0
    else:
        return 0


def parse_reddit(comment: str) -> str:
    # Parse a reddit comment with the !quant_bot flag and produce a response
    query_list = re.split("!quant_bot", comment)
    query_list.pop(0)
    query = query_list[0]
    query = query.lstrip()
    query = query.upper()
    method = query.split(' ')[0]
    if method == 'PEER' or method == 'PEERS':
        ticker = query.split(' ')[1]
        if good_ticker(ticker):
            my_message = peer_comp(ticker)
        else:
            my_message = '%s not a valid ticker\n\n' % ticker
    elif method == 'HIST' or method == 'HISTOGRAM':
        ticker = query.split(' ')[1]
        if good_ticker(ticker):
            my_message = ticker_histogram(ticker)
        else:
            my_message = '%s not a valid ticker\n\n' % ticker
    elif method == 'VOL' or method == 'VOLATILITY':
        ticker = query.split(' ')[1]
        if good_ticker(ticker):
            my_message = simple_vol(ticker)
        else:
            my_message = '%s not a valid ticker\n\n' % ticker
    else:
        my_message = ''
    return my_message


def test_submission(submission):
    # Iterate over posts in a subreddit looking for my !quant_bot flag, respond to the ones that have it.
    if re.search("!quant_bot", submission.selftext):
        if submission.id not in replied['comments'].tolist():
            my_message = parse_reddit(submission.selftext)
        else:
            my_message = ''
    else:
        my_message = ''
    bleep_bloop = '\n***\n^Made ^by ^/u/ikmckenz. ^For ^usage ^and ^issues ^see ^[GitHub](https://github.com/ikmckenz/quant_bot).'
    if len(my_message) > 0:
        my_message += bleep_bloop
        submission.reply(my_message)
        print("Replying to submission %s" % submission.id)
        sql = "insert into posts values(\'%s\');" % submission.id
        engine.execute(sql)
    # Reply to comments
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        if re.search("!quant_bot", comment.body):
            if comment.id not in replied['comments'].tolist():
                my_message = parse_reddit(comment.body)
            else:
                my_message = ''
        else:
            my_message = ''
        if len(my_message) > 0:
            my_message += bleep_bloop
            comment.reply(my_message)
            print("Replying to comment %s" % comment.id)
            sql = "insert into posts values(\'%s\');" % comment.id
            engine.execute(sql)


new_replies = pd.DataFrame(columns=['orig'])

subreddit = reddit.subreddit('testingground4bots+finance+wallstreetbets+stocks+investing')
for submission in subreddit.new(limit=50):
    test_submission(submission)

sql = "select distinct orig as comments from posts;"
replied = pd.read_sql(sql, engine)

for submission in subreddit.hot(limit=50):
    test_submission(submission)

