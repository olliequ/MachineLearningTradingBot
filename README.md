<p align="center" width="300">
  <img src="https://i.pinimg.com/originals/76/f3/ec/76f3ec2ea3bb788ae75fb099cf6e55a7.jpg" />
</p>

# Project Damianos: A ML-Reinforced Trading Bot

## Introduction
It's ill-advised to trade with emotion -- so why not have a bot do it for you? This project leverages the power of machine learning to devise an algorithm that provides advice on whether to purchase an asset at any given time. This advice is fed to a bot which automatically executes trades based on said advice.

**The problem:** It's an incredibly difficult task to manually derive correct-enough logic that gaurantees with enough confidence when to buy an asset, and when not to. One common practice is examining charts and cherry-picking certain instances where certain indicators result in a good buy -- this logic isn't neccessarily true for other time frames and carries dangerous bias. 

**The solution:** Use machine learning to identify and understand the relationships between various financial indicators over several thousand candles. If engineered correctly, the machine learning algorithm will develop a correlation between them (and other specified features) and identify a pattern, such that future candles and their associated features can be immediately analysed in real-time to assess whether or not they are classified as good buys.

## The Specifics
#### Summary 
This project is primarily powered by 2 Python scripts: `bot.py` and `botFunctions.py`. `bot.py` retrieves real-time data from a cryptocurrency exchange platform by establishing and maintaining a socket connection. The exchange is Binance, and the particular asset traded by default in this project is Ethereum. The incoming data is cleaned up and handed off to `botFunctions.py`. It is here where various financial indicators based on what the candle closes at are calculated and handed to a classifier which has been trained on historical data. In return, advice on whether to buy or not buy the candle is provided by the classifier. Analysed candles are actually appended to the historical dataset such that the classifier is continually trained.

#### Customizability
The bot is designed such that the following parameters are easily adjustable: 
- The particular asset traded (default is Ethereum)
- Candle length (default is 5 minutes)
- Feature choice (thus selection of financial indicators) 
- Classifier choice (default is Naïve Bayes)
- Label logic

#### Classifier Approach
At the end of the day, there’s only 2 actions the bot can take when a candle closes. Either buy right now, or don’t. One classifier implementation is using linear regression, where we train our data on different indicators (RSI, MACD, etc) which act as the features. The classifier will output a numerical value (the likelihood of the candle in question being a good buy). In this case, we'd set a threshold of whether to buy or not (e.g. if the probability spat out by the logistic regression function is higher than `0.85`, then the label assigned is `Buy`, and thus for this candle we place a buy order).

Another approach is using a classifier that instead of spitting out a number, it spits out a label which is either `Buy` or `Don't Buy` (so here it's spitting out a binary label, not a number -- this is thus called a *Binary Classifier*). If the label outputted for a candle is `Buy`, then we buy as the classifier believes that based on the feature correlation it has for the given candle, the price will go up. The classifier is first trained on a large *training data* set -- this involves giving it a dataset of candles **with their labels attached**. Then, once it's trained up, it's fed a test data set (a set of data (candles) which the bot hasn't seen -- thus these candles weren't in the training set), whereby the candles in this set *do* have attached labels, however these labels are **not** given to the bot, as the bot needs to calculate the labels itself -- this is the testing phase! If our bot is a success, then the labels outputted by the bot on the test instances (either `Buy` or `Don't Buy`) match the **true labels** of the data set (the ones we held back from the bot). If our bot is predicting the correct label most of the time, then this means it might be profitable and is a candidate for deployment. 

One question is asking what exactly constitutes a 'good buy' (i.e. when we're assigning labels to the candles in the dataset, for a given candle (row) how do we know whether to assign it a label of `Buy` or `Don't Buy`?)? We have to tell the bot for each candle (which remember has a unique set of features -- indicator values etc.) whether or not placing the buy order at that candle close is a good buy or not (if it is, then we assign it the `Buy` label). We could do this for Candle A by looking at a future candle relative to it -- if this candle that's *x* minutes into the future has a higher closing price, then this means Candle A *is* a good buy, and so we assign it the `Buy` label.

So to reiterate, the bot is fed different indicators, which are calculated based on the latest candle data (eg. RSI calculated over the last 14 candles). The bot takes all these in during the training phase, and identifies patterns between the indicators that indicate a buy signal. Apart from using financial indicators as features, we could additionally use other data like '*What was the price 30 minutes ago?*’ as a feature.

## Project State 
Because all intended goals are met, this project is now considered complete. With this said, as always there is still the option for future work. Areas of exploration include:

- Investigate performance of other classifiers
- Develop a website that live-tracks the bots actions and environment variables (e.g. account balance, current position, historical trades, PNL-to-date)

## Indicators Used

- **Relative Strength Index (RSI)**: The RSI is a *momentum indicator* that compares the number of days an instrument closes up versus closing down. These values are then ranged from 0 to 100, with overbuying typically expected when the RSI returns a value of over 70 and oversold securities are expected when the value is under 30. For example, if an instrument price reaches the upper band of a Bollinger Band price channel and, at the same time, the RSI reads 70+, you could discern that the instrument is overbought, and then sell the instrument.

- **Moving Average Convergence Divergence (MACD)**: MACD is a *trend-following momentum indicator* that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line," is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals (there is thus 2 lines for the MACD). Traders may buy the security when the MACD crosses above its signal line and sell—or short—the security when the MACD crosses below the signal line. Moving average convergence divergence (MACD) indicators can be interpreted in several ways, but the more common methods are crossovers, divergences, and rapid rises/falls.

- **Bollinger Bands**: Bollinger Bands are a set of trendlines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security's price, but which can be adjusted to user preferences. They're designed to discover opportunities that give investors a higher probability of properly identifying when an asset is oversold or overbought. There are three lines that compose Bollinger Bands: A simple moving average (middle band) and an upper and lower band. The upper and lower bands are typically 2 standard deviations +/- from a 20-day simple moving average, but these numerical values can be modified. When the Bollinger Bands are far apart, volatility is high. When they are close together, it is low. Bollinger Band Width (BBW, lol) is a measure of this 'gap'. Explanation: https://www.investopedia.com/articles/technical/04/030304.asp

- **On-Balance Volume (OBV)**: On-balance volume (OBV) is a *momentum (leading) indicator* that uses volume flow to predict changes in asset price. It's based on the belief that when volume increases sharply without a significant change in the stock's price, the price will eventually jump upward or fall downward. OBV shows crowd sentiment that can predict a bullish or bearish outcome. Comparing relative action between price bars and OBV generates more actionable signals than the green or red volume histograms commonly found at the bottom of price charts. Good explanation: https://www.investopedia.com/terms/o/onbalancevolume.asp.
