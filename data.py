import numpy as np
import pandas as pd
import talib

"""
The first thing we need to do is compute the desired features. That is, MACD, RSI, Bollinger etc.
We'll use the TA library to take in each candle (and previous candle closes before it) and calculate these
features. We then insert these values into the spreadsheet.
"""

rawData = pd.read_csv("./CSVs/ETHUSDT_15m_july1.csv") # Read in the original spreadsheet from TV as a Pandas dataframe.
rawData["RSI"] = 0.00 # Add a new column in the dataframe, whereby each value for each candle is initialised as 0 (this is obviously updated in the code)
rawData["MACD"] = 0.00 # Add a new column in the dataframe, MACD value initialised as 0.
rawData["MACDSIGNAL"] = 0.00 # Add a new column in the dataframe, MACD Signal value initialised as 0.
rawData["MACDHIST"] = 0.00 # Add a new column in the dataframe, MACD Hist value initialised as 0.
rawData["BBW"] = 0.00 # Bollinger Band Width -- a measure between the bands.
rawData["OBV"] = 0.00 # Add new column in the dataframe, OBV value initialised as 0.
rawData["Label"] = "No"
rawData.to_csv("./CSVs/withFeatures.csv", index=False) # Save this modified dataframe into a new CSV file.
withIndicators = pd.read_csv("./CSVs/withFeatures.csv") # Read in this new CSV file as a fresh dataframe, called 'withIndicators'.

RSI_Period = 14
Fast_Period = 12 # MACD fast period initialised at 12, subject to change.
Slow_Period = 26 # MACD slow period initialised at 26, subject to change.   
Signal_Period = 9 # MACD signal period initialised at 9, subject to change.

candleCloses = withIndicators.iloc[:,[0,4,7]].values # Isolate the candle's timestamp, its close value, and it's RSI value (initialised as 0
justCloses = withIndicators['close'].to_numpy() # "Close" column from CSV converted to numpy array
justVolume = withIndicators['Volume'].to_numpy() # "Volume" column from CSV converted to numpy array

"""
We want to iterate through each candle, and using its close value and the 13 candles before its close values, construct the RSI value for that candle.
"""

print(f"\n------\n***15M CANDLE DATA SET: Candles are from 01/07/21 to 04/02/22***\nSanity Check: The first row is: {candleCloses[0]}")
print(f"The number of rows/candle is: {len(withIndicators)}\n------")

# RSI
for i in range(0, len(withIndicators)): # Iterate over every single candle (row) in the dataset.
    if i >= 13:         # If the row we're on has 13 candles above it, we'll use those close values *and* its own close value to calculate its RSI. So, this doesn't happen for the first 13 rows (not enough candles yet)
        close_values = [candleCloses[i-13][1], candleCloses[i-12][1], candleCloses[i-11][1], candleCloses[i-10][1], candleCloses[i-9][1], candleCloses[i-8][1], candleCloses[i-7][1], candleCloses[i-6][1], candleCloses[i-6][1], candleCloses[i-5][1], candleCloses[i-4][1], candleCloses[i-3][1], candleCloses[i-2][1], candleCloses[i-1][1], candleCloses[i][1]]
        # print(f'The values used to calculate for RSI for this candle, {candleCloses[i][1]}, are: {close_values}') 
        np_closes = np.array(close_values) # Needs to be converted to a numpy array for TALIB to do the RSI calculation.
        rsi = talib.RSI(np_closes, RSI_Period)
        # print(rsi[-1]) # rsi is actually an array of size 14 elements, and so we need the last value of it, because the first 13 are NaN. 
        withIndicators["RSI"][i] = rsi[-1] # Insert this calculated value into the 'fresh' dataframe.


# MACD
macd, macdsignal, macdhist = talib.MACD(justCloses, Fast_Period, Slow_Period, Signal_Period) # MACD calculated using candle close data, Fast/Slow/Signal Periods can be changed at initialisation point.
withIndicators["MACD"]= macd # Calculated MACD values imported into CSV
withIndicators["MACDSIGNAL"] = macdsignal # Calculated MACD Signal values imported into CSV
withIndicators["MACDHIST"]= macdhist # Calculated MACD Hist values imported into CSV

# Bollinger Bands
upperband, middleband, lowerband = talib.BBANDS(justCloses, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
for i in range(0, len(middleband)):
    withIndicators["BBW"][i] = ((upperband[i]-lowerband[i])/middleband[i])

# OBV
real = talib.OBV(justCloses, justVolume)
withIndicators["OBV"] = real

print(f"\nBelow will simply print the first 20 candles. Open up withIndicators.csv to see them all. The first 13 candles below obviously don't have an RSI:\n\n{withIndicators.loc[0:19,:]}")
withIndicators.to_csv("./CSVs/withFeatures.csv", index=False) # The newly created CSV file (made on line 13) is overwritten with the inserted RSI values and saved.
