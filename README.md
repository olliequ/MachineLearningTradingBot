# Mr Botty!

## What functionality exists so far:
We have a script that retrieves real-time data from Binance (by establishing and maintaining a socket connection). This socket receives 'ticks' every second. 60 ticks constitutes a 1 minute candle (whose close value is the last tick). We're currently able to calculate various financial indicators based on what a candle closes at. Using this, we've written logic based off of these indicators -- e.g. `if RSI > 80, then buy 0.5 ETH`. 

**The problem:** We're not knowledgable enough to derive correct-enough logic that gaurantees with enough confidence when to buy, and when not to. All we've done is manually eye balled charts and cherry-picked certain instances where certain indicators result in a good buy -- this logic isn't neccessarily true for other time frames. 

**The solution:** Use Machine Learning (ie. Mr Botty) to identify and understand the relationships between the financial indicators. If we're lucky, the machine learning will develop a correlation between them and indetify a pattern between them, such that if the pattern is observed in future, it is a buy signal.

## The Specifics

At the end of the day, there’s one of 2 actions the bot takes when a candle closes. Either buy right now, or don’t. One method is using logistic regression, where we could train our data based on different indicators (rsi, macd, etc), then use that to predict the probability price going up or down. In this case, we'd set a threshold of which to buy or not (e.g. if the probability spat out by the logistic regression function is higher than 0.85, then place a buy order).

Another approach is using a classifier that instead of spitting out a number, it spits out a label which is either Buy or Dont Buy (so here it
s spitting out a binary label, not a number). If it’s ``Buy``, then you simply buy. Remember, we first have to train the bot on a large training data set. Then using what it has learned, we feed it a test data set (which the bot hasn't seen -- so it wasn't in the training set). Ideally, the labels (either Buy or Don't Buy) match the true labels of the data set -- which is whether or not that datapoint would've actually been a good buy (profitable), because that means it’s reliable and thus can be deployed to Binance. One question is asking what exactly constitues a 'good buy' (ie. when would a candle be given a true label of `Buy`)? We have to tell the bot for each candle (which has a unique set of features -- indicator values etc.) whether or not placing the buy order at that candle close is a good buy or not (if so, then we assign it the `Buy` label). You could do this by looking at the future price, like maybe 10 minutes after the candle instance, and if this candle that's 10 minutes into the future has a higher closing price, then yes it’s a good buying combination.

To reiterate, the bot is fed different indicators, which are calculated based on the latest candle data (eg RSI calculated on the last 14 ticks). And then the bot takes these all in and, using the training set phase, identifies patterns between which indicators combine to give a buy signal. Aside from feeding it financial indicators, we could also feed it other data like ‘what was the price 30 minutes ago’ or whatever. This is just another feature example we could add.

## ML Progress So Far

We now have a script `data.py` which takes in a CSV from TradingView (a bunch of candle closes over many months) and for each candle its respective RSI value is calculated. This value is inserted and saved into a new CSV file, so that the original one isn't overwritten. 

## Next Steps

Do the same thing for the other indicators we want calculatd for each candle (e.g MACD, Bollenger). After this, we will have a CSV that has each candle's respective feature values.

## Future Steps

Once the feature values are calculated for each row (candle), we launch the classifier we choose (e.g. Logistic Regression) that scans over the training set (the new CSV) and thus gets 'trained'. We then feed this trained classifier a new set of data, and it will tell us if each datapoint (each candle) is a good buy or not. We then see if each candle would've been a good buy or not, and hopefully the classifier we have got it right most of the time!

- Deploy Mr. Botty to Binance!
