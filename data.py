import numpy as np
import pandas as pd
import talib

"""
The first thing we need to do is compute the desired features. That is, MACD, RSI, Bollinger etc.
We'll use the Finance library to take in each candle (and it's previous closes) and calculate its
features. Then we'll insert it into its row using pandas insert.
"""
