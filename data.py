import numpy as np
import pandas as pd
import talib

"""
The first thing we need to do is compute the desired features. That is, MACD, RSI, Bollinger etc.
We'll use the Finance library to take in each candle (and it's previous closes) and calculate its
features. Then we'll insert it into its row using pandas insert.
"""

rawData = pd.read_csv("/Users/Ollie/Desktop/Ollie/Programming/Programming Projects/Trading_Bot/MachineLearningBot/ETHUSDT_15m_july1.csv") # Read in the original spreadsheet from TV as a Pandas dataframe.
rawData["RSI"] = 0.00 # Add a new column in the dataframe, whereby each value for each candle is initialised as 0 (this is obviously updated in the code)
rawData.to_csv("/Users/Ollie/Desktop/Ollie/Programming/Programming Projects/Trading_Bot/MachineLearningBot/RSI.csv", index=False) # Save this modified dataframe into a new CSV file.
withIndicators = pd.read_csv("/Users/Ollie/Desktop/Ollie/Programming/Programming Projects/Trading_Bot/MachineLearningBot/RSI.csv") # Read in this new CSV file as a fresh dataframe, called 'withIndicators'.

RSI_Period = 14

candleCloses = withIndicators.iloc[:,[0,4,7]].values
"""
We want to iterate through each candle, and using its close value and the 13 candles before its close values, construct the RSI value for that candle.
"""
print(candleCloses[0])
print(len(withIndicators))
for i in range(0, len(withIndicators)): # Iterate over every single candle (row) in the dataset.
    if i >= 13:         # If the row we're on has 13 candles above it, we'll use those close values *and* its own close value to calculate its RSI. So, this doesn't happen for the first 13 rows (not enough candles yet)
        close_values = [candleCloses[i-13][1], candleCloses[i-12][1], candleCloses[i-11][1], candleCloses[i-10][1], candleCloses[i-9][1], candleCloses[i-8][1], candleCloses[i-7][1], candleCloses[i-6][1], candleCloses[i-6][1], candleCloses[i-5][1], candleCloses[i-4][1], candleCloses[i-3][1], candleCloses[i-2][1], candleCloses[i-1][1], candleCloses[i][1]]
        # print(f'The values used to calculate for RSI for this candle, {candleCloses[i][1]}, are: {close_values}') 
        np_closes = np.array(close_values) # Needs to be converted to a numpy array for TALIB to do the RSI calculation.
        rsi = talib.RSI(np_closes, RSI_Period)
        # print(rsi[-1]) # rsi is actually an array of size 14 elements, and so we need the last value of it, because the first 13 are NaN. 
        withIndicators["RSI"][i] = rsi[-1] # Insert this calculated value into the 'fresh' dataframe.

print(withIndicators.loc[0:20,:])
withIndicators.to_csv("/Users/Ollie/Desktop/Ollie/Programming/Programming Projects/Trading_Bot/MachineLearningBot/RSI.csv", index=False) # The newly created CSV file (made on line 13) is overwritten with the inserted RSI values and saved.
