import numpy as np
import pandas as pd
import talib, copy, subprocess, platform, os, math, websocket, json, config, pprint
from collections import Counter
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

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

# connect to the binance stream. stream we're feeding off is the 1minute candles for the ETH/USDT pair
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"

TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.005 # amount of ETH purchased & sold per action
in_position = False             # variable for if we have already made a buy action or not.

client = Client(config.API_KEY, config.API_SECRET)  # make a client 'object' that just represents our personal binance account.

# framework for the 'order' function. this function is called when a buy or sell is needed. 4 arguments.
def order(side, quantity, symbol, order_type=ORDER_TYPE_MARKET):
    try:
        print("sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False
    
    return True

# prints message when a connection is first established to the binance websocket stream (i.e. when the program is started)
def on_open(ws):
    print("\nOpened connection - let's go!\n")
    print(config)

# prints message when a connection is closed off to the binance stream (i.e. when the program is ended)
def on_close(ws):
    print("Closed connection. See ya next time.\n")

# main function. this function is called everytime a candle tick is sent from the binance stream to us (so every 2 seconds... there are 30 ticks per 1m candle)
def on_message(ws, message):
    global closes, in_position     # global variables from above that we reference in this function.

    print("--------New candle tick inbound!--------")         
    json_message = json.loads(message) # takes json candle tick stream data (called `message`... comes every 2 seconds) and converts it to python data structure that is more useful
    pprint.pprint(json_message)      # uncomment the left to see the printing of each candle tick in the terminal (every 2 seconds)

    candle = json_message['k']         # each candle has three components (can see in binance API docs). we want the third one, denoted by 'k'
    is_candle_closed = candle['x']     # easy reference to whether or not the candle closed. One of the 30 ticks per minute will have this Boolean value as "True"
    close = candle['c']                # easy reference to what the closing price is (we are interested in the 'c' of the tick which has the closed value above as "True")
    high = candle['h']
    low = candle['l']
    volume = candle['v']

    if is_candle_closed:               # if the tick we're looking at is the 1 in 30 that is closed.
        print("This candle closed at {}.".format(close))
        justCloses.append(float(close))    # append its value to the list
        justHighs.append(float(high))    # append its value to the list
        justLows.append(float(low))    # append its value to the list
        justVolume.append(float(volume))    # append its value to the list
        # withIndicators['close'] = justCloses
        trainFeatures, trainLabels, testFeatures, testLabels = getFeaturesAndLabels(justLows, justHighs, justCloses, justVolume)

# Lines needed for the Binance data stream (called a websocket). Last line makes the stream run continuously.
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()

def getFeaturesAndLabels(justLows, justHighs, justCloses, justVolume):
    # RSI
    rsi = talib.RSI(justCloses, RSI_Period) 
    # withIndicators['RSI'] = rsi

    # MACD
    macd, macdsignal, macdhist = talib.MACD(justCloses, Fast_Period, Slow_Period, Signal_Period) # MACD calculated using candle close data, Fast/Slow/Signal Periods can be changed at initialisation point.
    # withIndicators["MACD"]= macd # Calculated MACD values imported into CSV
    # withIndicators["MACDSIGNAL"] = macdsignal # Calculated MACD Signal values imported into CSV
    # withIndicators["MACDHIST"]= macdhist # Calculated MACD Hist values imported into CSV

    # Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(justCloses, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bbw = ((upperband - lowerband) / middleband)
    # withIndicators["BBW"] = ((upperband - lowerband) / middleband)

    # OBV
    real = talib.OBV(justCloses, justVolume)
    # withIndicators["OBV"] = real

    # Stochastic
    slowk, slowd = talib.STOCH(justHighs, justLows, justCloses, fastk_period=14, slowk_period=1, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # withIndicators['SLOWK'] = slowk
    # withIndicators['SLOWD'] = slowd

    latestData = {'MACD': macd, 'MACDSIGNAL': macdsignal,'MACDHIST': macdhist, 'BBW': bbw, 'OBV': real, 'SLOWK': slowk, 'SLOWD': slowd}
    latestData["Label"] = 0 

    # Creating the labels and saving the completed dataset. 
    latestData["Label"] = np.where((withIndicators['close'] + 15) < withIndicators['close'].shift(-1), 1, 0)
    latestData.to_csv("./CSVs/withFeatures.csv", index=True, header=True) # The newly created CSV file (made on line 23) is overwritten with the features and labels inserted.
    # The below opens the CSV in Excel for viewing when this script is run.
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', "./CSVs/withFeatures.csv"))
    elif platform.system() == 'Windows':    # Windows
        os.startfile(".\CSVs\withFeatures.csv")

    # Now the data processing begins. We drop the useless columns and rows.
#    withIndicators.drop(    # Drop the columns that aren't features.
#            labels = ["time", "open", "high", "low", "close", "Volume", "Volume MA", "SLOWK", "SLOWD"],
#            axis = 1,
#            inplace = True
#            )
    latestData.drop(    # Drop the first 33 rows as they contain blank cells.
            labels = range(0, 33),
            axis=0,
            inplace = True
            )

    scaler = MinMaxScaler()
    withIndicators[["BBW", "OBV"]] = scaler.fit_transform(withIndicators[["BBW", "OBV"]])
    numberOfFeatures = withIndicators.shape[1]
    numberOfCandles = withIndicators.shape[0]
    """
    Below we partition the whole dataset into 4 parts: training data, training labels, test data, test labels.
    It's 90/10 split -- 90% of candles are used for training, and the other 10% is the test set where make the
    predictions and then compare them to the actual test labels.
    """
    print(f"There are {numberOfFeatures} columns, and {numberOfCandles} rows.")
    trainFeaturesDF = withIndicators.iloc[:math.floor(0.8*numberOfCandles), 0:numberOfFeatures-1]
    trainFeaturesDF.to_csv("./CSVs/trainFeatures.csv", index=False, header=False) # Save to a CSV so we can manually eyeball data.
    trainLabelsDF = withIndicators.iloc[:math.floor(0.8*numberOfCandles), numberOfFeatures-1]
    trainLabelsDF.to_csv("./CSVs/trainLabels.csv", index=False, header=False)
    testFeaturesDF = withIndicators.iloc[math.floor(0.8*numberOfCandles+1):, 0:numberOfFeatures-1]
    testFeaturesDF.to_csv("./CSVs/testFeatures.csv", index=False, header=False)
    testLabelsDF = withIndicators.iloc[math.floor(0.8*numberOfCandles+1):, numberOfFeatures-1]
    testLabelsDF.to_csv("./CSVs/testLabels.csv", index=False, header=False)

    featureNames = list(trainFeaturesDF.columns.values) 
    print(f"The features currently selected for training are: {featureNames}")
    trainFeatures = trainFeaturesDF.to_numpy() # Convert the partitioned dataframes above into numpy arrays (needed for the classifiers).
    trainLabels = trainLabelsDF.to_numpy()
    testLabels = testLabelsDF.to_numpy()
    testFeatures = testFeaturesDF.to_numpy()
    print("Dimensions of the partitioned dataframes:\n\t- trainFeatures: {}\n\t- trainLabels: {}\n\t- testFeatures: {}\n\t- testLabels: {}".format(trainFeatures.shape, trainLabels.shape, testFeatures.shape, testLabels.shape))

    """
    Mutual Information (MI) measures the correlation of each feature with the labels; the degree as to which
    each feature affects the label. Optional metric, but I included it for fun.
    """
    highest_mi_feature_name = "" # feature with highest MI
    lowest_mi_feature_name = "" # feature with lowest MI
    mi_score = MIC(trainFeatures, trainLabels, discrete_features=False)
    index_of_largest_mi = np.argmax(mi_score)
    largest_mi = mi_score[index_of_largest_mi]
    index_of_lowest_mi = np.argmin(mi_score)
    lowest_mi = mi_score[index_of_lowest_mi]
    highest_mi_feature_name = featureNames[index_of_largest_mi]
    lowest_mi_feature_name = featureNames[index_of_lowest_mi]
    print(f"\nThe feature with the highest MI is: {highest_mi_feature_name}")
    print(f"The feature with the lowest MI is: {lowest_mi_feature_name}")
    print(f'All 16 MIC scores are as follows: {np.round(mi_score, 3)}')

    """
    Now that the CSV is cleaned up and we have an idea of MI, we can begin implementing the classifier.
    """
    print("\n------\nData is now cleaned up and partioned, so let's apply the classifiers.\n------\n")

# ---> Naive Bayes
def NB_Classifier(train_features, train_labels, test_features): # Naive Bayers classifier.
    predictions = []
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    predictions = gnb.predict(test_features)
    return predictions

nb_predictions = NB_Classifier(trainFeatures, trainLabels, testFeatures) # Predictions for the test set. 
print(f"1) Naive Bayes\n\t- Predicted class distribution:\t- {Counter(nb_predictions)}")
NB_error = 1 - accuracy_score(nb_predictions, testLabels)
NB_f1 = f1_score(nb_predictions, testLabels, average='macro')
print(f"\n---> NB\t\tError: {round(NB_error, 2)}\tMacro F1: {round(NB_f1, 2)}")

testCloses = justCloses[16770:]
dict = {'closes': testCloses, 'actual': testLabels, 'predicted': nb_predictions}
df_ = pd.DataFrame(dict)
df_.to_csv('hmmm.csv')

profit_loss = 0
for i in range(len(nb_predictions)-1):
    if nb_predictions[i] == 1:
        profit_loss = profit_loss + (testCloses[i+1]-testCloses[i])
print(f"Outcome is: ${round(profit_loss, 2)}")

# ---> Logistic Regression
lr_predictions = []
trainFeaturesNormalised = []
testFeaturesNormalised = []

mean = np.mean(trainFeatures, axis=0)
std = np.std(trainFeatures, axis=0)
np.set_printoptions(suppress=True)

def normalizer(array):
    normalized_array = copy.deepcopy(array)
    for i in range(0, len(array[0])):
        for j in range(0, len(array)):
            normalized_array[j][i] = (array[j][i] - mean[i])/std[i]  
    return normalized_array

trainFeaturesNormalised = normalizer(trainFeatures)
testFeaturesNormalised = normalizer(testFeatures)

logisticRegr = LogisticRegression()
logisticRegr.fit(trainFeaturesNormalised, trainLabels)
lr_predictions = logisticRegr.predict(testFeaturesNormalised)

score = logisticRegr.score(testFeaturesNormalised, testLabels)
lr_err = 1 - accuracy_score(lr_predictions, testLabels)
lr_f1 = f1_score(lr_predictions, testLabels, average='macro')

print(f"\n2) Logistic Regression\n\t- Predicted class distribution:\t{Counter(lr_predictions)}")
print(f'\t- The coefficients for this LR classifier are:\t{logisticRegr.coef_}')
print(f"\n---> LR\t\tError: {round(lr_err, 2)}\tMacro F1: {round(lr_f1, 2)}")

"""
Current problem: Both classifiers are predicting 0 as the label for every candle. 
83% is the given accuracy only because the test labels are 83% 0. Thus, it's a misleading accuracy.
"""

