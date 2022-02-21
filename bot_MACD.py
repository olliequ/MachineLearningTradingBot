# import neccessary libraries
import websocket, json, talib, numpy
import config
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

# connect to the binance stream. stream we're feeding off is the 1minute candles for the ETH/USDT pair
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"

fastperiod = 12
slowperiod = 26
signalperiod = 9
TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.005 # amount of ETH purchased & sold per action

closes = []                     # starts off as an empty list. each time a candle closes, its value gets appended to this list. list resets when program is terminated.
in_position = False             # variable for if we have already made a buy action or not.
nine_above_twenty = True
just_crossed = False
nine_above_twenty_counter = 0   # how many successive candles has the nine been above the twenty for?
twenty_above_nine_counter = 0   # how many successive candles has the twenty been above the nine for?

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
    global closes, in_position, nine_above_twenty, twenty_above_nine_counter, nine_above_twenty_counter     # global variables from above that we reference in this function.

    print("--------New candle tick inbound!--------")         
    json_message = json.loads(message) # takes json candle tick stream data (called `message`... comes every 2 seconds) and converts it to python data structure that is more useful
    # pprint.pprint(json_message)      # uncomment the left to see the printing of each candle tick in the terminal (every 2 seconds)

    candle = json_message['k']         # each candle has three components (can see in binance API docs). we want the third one, denoted by 'k'
    is_candle_closed = candle['x']     # easy reference to whether or not the candle closed. One of the 30 ticks per minute will have this Boolean value as "True"
    close = candle['c']                # easy reference to what the closing price is (we are interested in the 'c' of the tick which has the closed value above as "True")

    if is_candle_closed:               # if the tick we're looking at is the 1 in 30 that is closed.
        print("This candle closed at {}.".format(close))
        closes.append(float(close))    # append its value to the list
        print("All candle closes:")
        print(closes)                  # print the entire list for our reference
        print("")

        if len(closes) > slowperiod - 1:                      # if the list contains over 20 values we can start working. 20 are needed because ema_twenty needs 20 values to calculate itself.
            np_closes = numpy.array(closes)                     # convert list to numpy array. needed for math purposes for the TA library
            macd, macdsignal, macdhist = talib.MACD(np_closes, fastperiod, slowperiod, signalperiod)        # call the EMA function from talib to do the calc for us. the result is an array, which contains all previously calculated values... run program to understand what I mean
            print("--!! 3 MACD Values Calculated !!--\n")
            print("All MACD's calculated so far:\n")
            print(macd)
            print("\nAll macdhist's calculated so far:\n")
            print(macdhist)
            print("\nAll signal's calculated so far:\n")
            print(macdsignal)
            last_macd = macd[-1]                    # most recent value of the array
            last_macdhist = macdhist[-1]
            print("\nThe current macd is {}".format(last_macd))
            print("The current macdhist is {}\n".format(last_macdhist))

            # if last_ema_nine >= last_ema_twenty:            # logic for the counters... I made the counters as a way to track how long (how many candles), for example, the nine has been above the twenty for
            #     nine_above_twenty_counter += 1
            #     twenty_above_nine_counter = 0
            # elif last_ema_nine < last_ema_twenty:
            #     twenty_above_nine_counter += 1
            #     nine_above_twenty_counter = 0

            if (ema_nine[-1] < ema_twenty[-1]) and (ema_nine[-2] > ema_twenty[-2]):     # if hist < 0.3 (after prev crossing above 0)
                if in_position:                                                         # below only executes if we have some to sell
                    print("9 just fell below the 20! SELLL!")
                    # put binance sell logic here
                    order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)    # this is the line where the order is actually made in our account. the 'order' function from above is called.
                    if order_succeeded:
                        in_position = False                                             # sets our global Boolean variables to False. Means if this logic is called straight after, nothing will happen (line 87)
                else: 
                    print("The 9 just fell below the 20 and we would sell, but we don't own any, so nothing to do.")

            if (last_ema_twenty > 1.0070 * last_ema_nine): # if breached B-low && hist < -1.8 ... if 25 ticks pass and hist !>1, then sell
                if in_position:
                    print("It's looking like it will rise, but you already own it, so just sit tight and pray, lol.")
                else:
                    print("The 20 is way above the 9, indicating the price will go up! BUUUUY!")
                    # put binance buy logic here
                    order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = True
            print("-------This candle has finished processing, and the 9 & 20 for it have been compared. Onto the next one!-------\n")

# Lines needed for the Binance data stream (called a websocket). Last line makes the stream run continuously.
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()
