# Mr Botty!

## What functionality exists so far:
We have a script that retrieves real-time data from Binance (by establishing and maintaining a socket connection). This socket receives 'ticks' every second. 60 ticks constitutes a 1 minute candle (whose close value is the last tick). We're currently able to calculate various financial indicators based on what a candle closes at. Using this, we've written logic based off of these indicators -- e.g. `if RSI > 80, then buy 0.5 ETH`. 

**The problem:** We're not knowledgable enough to derive correct-enough logic that gaurantees with enough confidence when to buy, and when not to. All we've done is manually eye balled charts and cherry-picked certain instances where certain indicators result in a good buy -- this logic isn't neccessarily true for other time frames. 

**The solution:** Use Machine Learning (ie. Mr Botty!) to identify and understand the relationships between the financial indicators over several thousand candles. If we're lucky, the machine learning will develop a correlation between them and identify a pattern between them, such that if the pattern is observed in future, it is a buy signal.

## The Specifics

At the end of the day, there’s one of 2 actions the bot takes when a candle closes. Either buy right now, or don’t. One implementation is using Linear Regression, where we train our data on different indicators (RSI, MACD, etc) which are the features. The classifier will output a numerical value (the likelihood of the candle in question being a good buy). In this case, we'd set a threshold of where to buy or not (e.g. if the probability spat out by the logistic regression function is higher than `0.85`, then the label assigned is `Buy`, and thus for this candle place a buy order).

Another approach is using a classifier that instead of spitting out a number, it spits out a label which is either `Buy` or `Don't Buy` (so here it's spitting out a binary label, not a number -- this is called a Binary Classifier). If the label outputted for a candle is `Buy`, then you simply buy. Remember, we first have to train the bot on a large training data set -- this involves giving it a dataset of candles **with their labels attached**. Then, once it's trained up, we feed it a test data set (which the bot hasn't seen -- so it wasn't in the training set), whereby the candles in this set have attached labels however these labels are **not** given to the bot. If our bot is a success, then the labels outputted by the bot on the test instances (either `Buy` or `Don't Buy`) match the **true labels** of the data set (the ones we held back from the bot). These labels indicate whether a candle would've actually been a good buy (profitable), and so if our is predicting the correct label, then this means it’s profitable and thus can be deployed to Binance. 

One question is asking what exactly constitues a 'good buy' (ie. when we're assigning labels to the candles in the dataset, how do we know whether to assign a label of `Buy` or `Don't Buy`)? We have to tell the bot for each candle (which has a unique set of features -- indicator values etc.) whether or not placing the buy order at that candle close is a good buy or not (if it is, then we assign it the `Buy` label). We could do this for Candle A by looking at a future candle, perhaps the candle that is 10 minutes after Candle A. If this candle that's 10 minutes into the future has a higher closing price, then it means Candle A is a good buy, and so we assign it the `Buy` label.

To reiterate, the bot is fed different indicators, which are calculated based on the latest candle data (eg. RSI calculated on the last 14 candles). The bot takes these all in during the training set phase, and identifies patterns between the indicators that give a buy signal. Aside from using financial indicators as features, we could additionally use other data like '*What was the price 30 minutes ago?*’ as a feature. This is just another feature example we could add.

## ML Progress So Far

We now have a script `data.py` which takes in a CSV from TradingView (a bunch of candle closes over many months) and for each candle (15m) its respective RSI and MACD value is calculated. This value is inserted and saved into a new CSV file (in the repo it's currently called `withFeatures.csv`), so that the original one isn't overwritten. 

## Next Steps

Do the same thing for the other indicators we want calculatd for each candle (e.g MACD, Bollinger Bands). After this, we will have a CSV that has each candle's respective feature values.

## Future Steps

- Once the feature values are calculated for each row (candle), we need to assign each candle a label (either `Buy` or `Don't Buy`), which can be done using the technique discussed above

- We then launch the classifier of choice (e.g. Logistic Regression) that scans over the training set (the new CSV) and thus gets 'trained'. We then feed this trained classifier a new set of data (the test data set), and it will tell us if each datapoint (each candle) is a good buy or not. We then compare the outputted labels by the classifier to the *actual* labels of the test data set, and hopefully the classifier got it right most of the time!

- Do the same for the 1m candle data, and the 30m candle data.

- Deploy Mr. Botty to Binance!

## Extra Notes:

- For a 15m candle dataset, the 06:45 candle opens at a price, and this is the price that 06:44:59 ticks over into. Thus, the price the 06:45 candle closes at is the opening price of the 07:00 candle (essentially the price at 06:59:59). When a candle closes, we immediately calculate the RSI using the last 14 candle closes (which includes the current candle's closing price). We then act depending on the value. So, we want to loop through each candle (row), and using its close value and the previous 13 ones, calculate the RSI for that candle. That is, the RSI value as a result of that candle’s close.
- Everytime we implement a new indicator and fill out the csv, it would be wise to verify they are the correct values by checking it against the actual chart on Binance. For example seeing that the 15m candle at July 1st 06:45 does indeed have an RSI value of 23.1 on the Binance chart
