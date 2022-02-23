import numpy as np
import pandas as pd
from datetime import datetime
import talib, copy, subprocess, platform, os, math, websocket, json, config, pprint, botFunctions
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

"""
The first thing we need to do is compute the desired features. That is, MACD, RSI, Bollinger etc.
We'll use the TA library to take in each candle (and previous candle closes before it) and calculate these
features. We then insert these values into the spreadsheet.
"""

rawData = pd.read_csv("./CSVs/Historical Data/eth_5m_feb23.csv") # Read in the original spreadsheet from TV as a Pandas dataframe.
rawData["RSI"] = 0.00           # Add a new column in the dataframe, whereby each value for each candle is initialised as 0 (this is obviously updated in the code)
rawData["MACD"] = 0.00          # Add a new column in the dataframe, MACD value initialised as 0.
rawData["MACDSIGNAL"] = 0.00    # Add a new column in the dataframe, MACD Signal value initialised as 0.
rawData["MACDHIST"] = 0.00      # Add a new column in the dataframe, MACD Hist value initialised as 0.
rawData["BBW"] = 0.00           # Bollinger Band Width -- a measure between the bands.
rawData["OBV"] = 0.00           # Add new column in the dataframe, OBV value initialised as 0.
rawData["SLOWK"] = 0.00
rawData["SLOWD"] = 0.00
rawData["Label"] = 0 
rawData.to_csv("./CSVs/Script Outputs/withFeatures.csv", index=False)  # Save this modified dataframe into a new CSV file.
withIndicators = pd.read_csv("./CSVs/Script Outputs/withFeatures.csv") # Read in this new CSV file as a fresh dataframe, called 'withIndicators'.

justCloses = withIndicators['close'].to_numpy().tolist()      # "Close" column from CSV converted to numpy array
justVolume = withIndicators['Volume'].to_numpy().tolist()     # "Volume" column from CSV converted to numpy array
justHighs = withIndicators['high'].to_numpy().tolist()        # "High" column from CSV converted to numpy array
justLows = withIndicators['low'].to_numpy().tolist()          # "Low" column from CSV converted to numpy array

ts_start = datetime.utcfromtimestamp(int(rawData["time"].iloc[0])).strftime('%Y-%m-%d %H:%M:%S')
ts_end = datetime.utcfromtimestamp(int(rawData["time"].iloc[-1])).strftime('%Y-%m-%d %H:%M:%S')
print(f"\n------\n***5m CANDLE DATA SET: Candles are from {ts_start} to {ts_end}***\nThe total number of rows/candles is: {len(withIndicators)}\n------\n")

# Before we open the socket connection, we'll first do a test run on the historic data we have, just for reference.
trainFeatures, trainLabels, testFeatures, testLabels = botFunctions.getFeaturesAndLabels(justLows, justHighs, justCloses, justVolume)
predictions = botFunctions.NB_Classifier(trainFeatures, trainLabels, testFeatures, testLabels) 
testCloses = np.array(justCloses[17034:])   # Extra 33 as 33 rows are shaved off in the bot function (which isn't called yet here).
print(f"\nThe lengths of the test set's closes, labels, and prediction labels are: {testCloses.shape} | {testLabels.shape} | {predictions.shape}\n")
dict = {'closes': testCloses, 'actual': testLabels, 'predicted': predictions}
df_ = pd.DataFrame(dict)
df_.to_csv('./CSVs/Script Outputs/LabelsComparison.csv')
botFunctions.profitLoss(testCloses, predictions)
print("\n------------\nThe historical data has been analysed with a suitable classifier selected. Time to go online and start trading!\n------------")

# Connect to the binance stream. stream we're feeding off is the 1minute candles for the ETH/USDT pair.
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_5m"

TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.02          # Amount of ETH purchased & sold per action.
in_position = False             # Variable for if we have already made a buy action or not.

client = Client(config.API_KEY, config.API_SECRET)  # Make a client 'object' that just represents our personal binance account.

# Framework for the 'order' function. this function is called when a buy or sell is needed. 4 arguments.
def order(side, quantity, symbol, order_type=ORDER_TYPE_MARKET):
    try:
        print("---> Sending order to Binance")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False
    
    return True

# Prints a message when a connection is first established to the binance websocket stream (i.e. when the program is started).
def on_open(ws):
    print("\n***Connection to exchange successful***\n")
    print(config)

# Prints a message when a connection is closed off to the binance stream (i.e. when the program is ended).
def on_close(ws):
    print("\n***Socket connection successfully closed***\n")

# Main function. This function is called everytime a candle tick is sent from the binance stream to us (so every 2 seconds... there are 30 ticks per 1m candle)
def on_message(ws, message):
    global justLows, justHighs, justCloses, justVolume, in_position     # Global variables from above that we reference in this function.

    print("--------New candle tick received!--------")         
    json_message = json.loads(message) # Takes json candle tick stream data (called `message`... comes every 2 seconds) and converts it to python data structure that is more useful
    # pprint.pprint(json_message)      # Uncomment this to see the printing of each candle tick in the terminal (every 2 seconds) and all its respective data.

    candle = json_message['k']         # Each candle has three components (can see in binance API docs). we want the third one, denoted by 'k'
    is_candle_closed = candle['x']     # Easy reference to whether or not the candle closed. One of the 30 ticks per minute will have this Boolean value as "True"
    close = candle['c']                # Easy reference to what the closing price is (we are interested in the 'c' of the tick which has the closed value above as "True")
    high = candle['h']
    low = candle['l']
    volume = candle['v']

    if is_candle_closed:               # If the tick we're looking at is the one that is closed.
        print(f"This candle closed at {round(float(close), 2)}. Let's feed it to the classifier and see what it suggests to do.\n")
        justLows.append(float(low))
        justHighs.append(float(high))
        justCloses.append(float(close))
        justVolume.append(float(volume))
        trainFeatures, trainLabels, testFeatures, testLabels = botFunctions.getFeaturesAndLabels(justLows, justHighs, justCloses, justVolume)
        predictions = botFunctions.NB_Classifier(trainFeatures, trainLabels, testFeatures, testLabels) 

        testCloses = justCloses[17034:]
        print(len(testCloses), len(predictions), len(testLabels))   # Checking that all 3 arrays are equal size.
        dict = {'closes': testCloses, 'actual': testLabels, 'predicted': predictions}
        df_ = pd.DataFrame(dict)
        df_.to_csv('./CSVs/Script Outputs/LabelsComparison.csv')
        botFunctions.profitLoss(testCloses, predictions)

        if predictions[-1] == 1 and not in_position:    # Bot has predicted it's a good buy; if we're not in position then place a buy order at this candle.
            print("---> Bot thinks we should buy!")
            # order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)     # Order is actually made in our account. the 'order' function from above is called.
            order_succeeded = True
            if order_succeeded:
                print(f"Purchased ETH at {round(justCloses[-1], 2)}.")
                in_position = True
            else: 
                print("---> Something went wrong during purchase.")

        elif predictions[-1] == 1 and in_position:
            print("---> We're already in position so despite this oppurtunity we won't buy again -- we only hold 1 unit at a time.")

        elif predictions[-1] == 0 and in_position:      # If we're in position, only if the bot doesn't say the current candle is a good buy do we sell.
            # order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)    
            order_succeeded = True
            if order_succeeded:
                print(f"---> ETH sold at {round(justCloses[-1], 2)}.")
                in_position = False
            else:
                print("***Something went wrong during sell***")

        elif predictions[-1] == 0 and not in_position:
            print("---> Bot thinks we shouldn't buy")

# Lines needed for the Binance data stream (called a websocket). Last line ensures the stream run continuously.
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()
