import numpy as np
import pandas as pd
import talib, copy, subprocess, platform, os 
from collections import Counter
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
rawData["SLOWK"] = 0.00
rawData["SLOWD"] = 0.00
rawData["Label"] = 0
rawData.to_csv("./CSVs/withFeatures.csv", index=False) # Save this modified dataframe into a new CSV file.
withIndicators = pd.read_csv("./CSVs/withFeatures.csv") # Read in this new CSV file as a fresh dataframe, called 'withIndicators'.

RSI_Period = 14
Fast_Period = 12 # MACD fast period initialised at 12, subject to change.
Slow_Period = 26 # MACD slow period initialised at 26, subject to change.   
Signal_Period = 9 # MACD signal period initialised at 9, subject to change.

justCloses = withIndicators['close'].to_numpy() # "Close" column from CSV converted to numpy array
justVolume = withIndicators['Volume'].to_numpy() # "Volume" column from CSV converted to numpy array
justHighs = withIndicators['high'].to_numpy() # "High" column from CSV converted to numpy array
justLows = withIndicators['low'].to_numpy() # "Low" column from CSV converted to numpy array

print(f"\n------\n***15m CANDLE DATA SET: Candles are from 01/07/21 to 04/02/22***\nThe total number of rows/candles is: {len(withIndicators)}\n------\n")

# RSI
rsi = talib.RSI(justCloses, RSI_Period) 
withIndicators['RSI'] = rsi

# MACD
macd, macdsignal, macdhist = talib.MACD(justCloses, Fast_Period, Slow_Period, Signal_Period) # MACD calculated using candle close data, Fast/Slow/Signal Periods can be changed at initialisation point.
withIndicators["MACD"]= macd # Calculated MACD values imported into CSV
withIndicators["MACDSIGNAL"] = macdsignal # Calculated MACD Signal values imported into CSV
withIndicators["MACDHIST"]= macdhist # Calculated MACD Hist values imported into CSV

# Bollinger Bands
upperband, middleband, lowerband = talib.BBANDS(justCloses, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
withIndicators["BBW"] = ((upperband - lowerband) / middleband)

# OBV
real = talib.OBV(justCloses, justVolume)
withIndicators["OBV"] = real

# Stochastic
slowk, slowd = talib.STOCH(justHighs, justLows, justCloses, fastk_period=14, slowk_period=1, slowk_matype=0, slowd_period=3, slowd_matype=0)
withIndicators['SLOWK'] = slowk
withIndicators['SLOWD'] = slowd

# Creating the labels and saving the completed dataset. 
withIndicators["Label"] = np.where((withIndicators['close'] + 10 ) < withIndicators['close'].shift(-1), 1, 0)
withIndicators.to_csv("./CSVs/withFeatures.csv", index=True, header=True) # The newly created CSV file (made on line 23) is overwritten with the features and labels inserted.
# The below opens the CSV in Excel for viewing when this script is run.
if platform.system() == 'Darwin':       # macOS
    subprocess.call(('open', "./CSVs/withFeatures.csv"))
#elif platform.system() == 'Windows':    # Windows
    #os.startfile(".\CSVs\withFeatures.csv")

# Now the data processing begins. We drop the useless columns and rows.
withIndicators.drop(    # Drop the columns that aren't features.
        labels = ["time", "open", "high", "low", "Volume MA"],
        axis = 1,
        inplace = True
        )
withIndicators.drop(    # Drop the first 33 rows as they contain blank cells.
        labels = range(0, 33),
        axis=0,
        inplace = True
        )

withIndicators.to_csv("./CSVs/data.csv", index=False)

features = withIndicators[['RSI', 'SLOWK', 'MACDHIST', 'MACD', 'OBV', 'SLOWD', 'MACDSIGNAL', 'Volume']]
labels = withIndicators['close']
features.to_csv("./CSVs/data.csv", index=False)
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

frame = pd.DataFrame(features)
frame.to_csv('./CSVs/scaled.csv')

x_train, x_test, y_train, y_test = train_test_split(features, labels)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict = lm.predict(x_test)

y_test = np.array(y_test)
print(y_test)
print(y_predict)

#print('F1 Score: ', f1_score(y_test, y_predict))


print("Train Score:")
print(lm.score(x_train, y_train))

print("Test Score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)
plt.plot(range(5000), range(5000))

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Eth Price vs Predicted Eth Price")

plt.show()


#print("Accuracy Score: ", accuracy_score(y_test, y_predict))

print(model.coef_)

#price_test = [[78.35, 81.34, 88.79]]
#price_test = scaler.fit_transform(price_test)
#predict = model.predict(price_test)
#print(predict)
