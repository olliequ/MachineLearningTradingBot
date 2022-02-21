import numpy as np
import pandas as pd
import talib, copy, subprocess, platform, os, math, websocket, json, config, pprint, botFunctions
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

"""
The first thing we need to do is compute the desired features. That is, MACD, RSI, Bollinger etc.
We'll use the TA library to take in each candle (and previous candle closes before it) and calculate these
features. We then insert these values into the spreadsheet.
"""

rawData = pd.read_csv("./CSVs/ETHUSDT_15m_july1.csv") # Read in the original spreadsheet from TV as a Pandas dataframe.
rawData["RSI"] = 0.00           # Add a new column in the dataframe, whereby each value for each candle is initialised as 0 (this is obviously updated in the code)
rawData["MACD"] = 0.00          # Add a new column in the dataframe, MACD value initialised as 0.
rawData["MACDSIGNAL"] = 0.00    # Add a new column in the dataframe, MACD Signal value initialised as 0.
rawData["MACDHIST"] = 0.00      # Add a new column in the dataframe, MACD Hist value initialised as 0.
rawData["BBW"] = 0.00           # Bollinger Band Width -- a measure between the bands.
rawData["OBV"] = 0.00           # Add new column in the dataframe, OBV value initialised as 0.
rawData["SLOWK"] = 0.00
rawData["SLOWD"] = 0.00
rawData["Label"] = 0 
rawData.to_csv("./CSVs/withFeatures.csv", index=False) # Save this modified dataframe into a new CSV file.
withIndicators = pd.read_csv("./CSVs/withFeatures.csv") # Read in this new CSV file as a fresh dataframe, called 'withIndicators'.

justCloses = withIndicators['close'].to_numpy()     # "Close" column from CSV converted to numpy array
justVolume = withIndicators['Volume'].to_numpy()    # "Volume" column from CSV converted to numpy array
justHighs = withIndicators['high'].to_numpy()       # "High" column from CSV converted to numpy array
justLows = withIndicators['low'].to_numpy()         # "Low" column from CSV converted to numpy array

print(f"\n------\n***15m CANDLE DATA SET: Candles are from 01/07/21 to 04/02/22***\nThe total number of rows/candles is: {len(withIndicators)}\n------\n")

# Before we open the socket connection, we'll first do a test run on the historic data we have, just for reference.
trainFeatures, trainLabels, testFeatures, testLabels = botFunctions.getFeaturesAndLabels(justLows, justHighs, justCloses, justVolume)
predictions = botFunctions.NB_Classifier(trainFeatures, trainLabels, testFeatures, testLabels) 
testCloses = justCloses[16770:]
print(testCloses.shape)
dict = {'closes': testCloses, 'actual': testLabels, 'predicted': predictions}
df_ = pd.DataFrame(dict)
df_.to_csv('hmmm.csv')
botFunctions.profitLoss(testCloses, predictions)

# Connect to the binance stream. stream we're feeding off is the 1minute candles for the ETH/USDT pair.
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"

TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.025          # Amount of ETH purchased & sold per action.
in_position = False             # Variable for if we have already made a buy action or not.

client = Client(config.API_KEY, config.API_SECRET)  # Make a client 'object' that just represents our personal binance account.

# Framework for the 'order' function. this function is called when a buy or sell is needed. 4 arguments.
def order(side, quantity, symbol, order_type=ORDER_TYPE_MARKET):
    try:
        print("sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False
    
    return True

# Prints message when a connection is first established to the binance websocket stream (i.e. when the program is started).
def on_open(ws):
    print("\nOpened connection - let's go!\n")
    print(config)

# Prints message when a connection is closed off to the binance stream (i.e. when the program is ended)
def on_close(ws):
    print("Closed connection. See ya next time.\n")

# Main function. This function is called everytime a candle tick is sent from the binance stream to us (so every 2 seconds... there are 30 ticks per 1m candle)
def on_message(ws, message):
    global justLows, justHighs, justCloses, justVolume, in_position     # Global variables from above that we reference in this function.

    print("--------New candle tick inbound!--------")         
    json_message = json.loads(message) # Takes json candle tick stream data (called `message`... comes every 2 seconds) and converts it to python data structure that is more useful
    # pprint.pprint(json_message)      # Uncomment the left to see the printing of each candle tick in the terminal (every 2 seconds)

    candle = json_message['k']         # Each candle has three components (can see in binance API docs). we want the third one, denoted by 'k'
    is_candle_closed = candle['x']     # Easy reference to whether or not the candle closed. One of the 30 ticks per minute will have this Boolean value as "True"
    close = candle['c']                # Easy reference to what the closing price is (we are interested in the 'c' of the tick which has the closed value above as "True")
    high = candle['h']
    low = candle['l']
    volume = candle['v']

    if is_candle_closed:               # if the tick we're looking at is the 1 in 30 that is closed.
        print("This candle closed at {}.".format(close))
        print("1")
        justCloses.append(close)
        justHighs.append(high)
        justLows.append(low)
        justVolume.append(volume)
        print("2")
        trainFeatures, trainLabels, testFeatures, testLabels = botFunctions.getFeaturesAndLabels(justLows, justHighs, justCloses, justVolume)
        predictions = botFunctions.NB_Classifier(trainFeatures, trainLabels, testFeatures, testLabels) 
        print("3")
        testCloses = justCloses[16770:]
        print(testCloses.shape)
        dict = {'closes': testCloses, 'actual': testLabels, 'predicted': predictions}
        df_ = pd.DataFrame(dict)
        df_.to_csv('hmmm.csv')
        botFunctions.profitLoss(testCloses, predictions)

        if predictions[-1] == 1 and not in_position:     # Bot has predicted it's a good buy; if we're not in position then place a buy order at this candle.
            order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
            if order_succeeded:
                in_position = True
                print(f"Purchased ETH at {justCloses}.")
            else: 
                print("Something went wrong during purchase.")
        elif in_position:
            order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)    # Order is actually made in our account. the 'order' function from above is called.
            if order_succeeded:
                in_position = False
                print(f"ETH sold at {justCloses}.")
            else:
                print("Something went wrong during sell.")
        elif predictions[-1] == 0:
            print("Don't buy!")

# Lines needed for the Binance data stream (called a websocket). Last line makes the stream run continuously.
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()
