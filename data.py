import numpy as np
import pandas as pd
import talib, copy, subprocess, platform, os 
from collections import Counter
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

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
withIndicators["Label"] = np.where((withIndicators['close'] + 10) < withIndicators['close'].shift(-1), 1, 0)
withIndicators.to_csv("./CSVs/withFeatures.csv", index=True, header=True) # The newly created CSV file (made on line 23) is overwritten with the features and labels inserted.
# The below opens the CSV in Excel for viewing when this script is run.
if platform.system() == 'Darwin':       # macOS
    subprocess.call(('open', "./CSVs/withFeatures.csv"))
elif platform.system() == 'Windows':    # Windows
    os.startfile("./CSVs/withFeatures.csv")

# Now the data processing begins. We drop the useless columns and rows.
withIndicators.drop(    # Drop the columns that aren't features.
        labels = ["time", "open", "high", "low", "close", "Volume", "Volume MA", "MACD", "MACDSIGNAL", "SLOWK", "SLOWD"],
        axis = 1,
        inplace = True
        )
withIndicators.drop(    # Drop the first 33 rows as they contain blank cells.
        labels = range(0, 33),
        axis=0,
        inplace = True
        )

"""
Below we partition the whole dataset into 4 parts: training data, training labels, test data, test labels.
It's 90/10 split -- 90% of candles are used for training, and the other 10% is the test set where make the
predictions and then compare them to the actual test labels.
"""
trainFeaturesDF = withIndicators.iloc[:18000, 0:4]
trainFeaturesDF.to_csv("./CSVs/trainFeaturesDF.csv", index=False, header=False) # Save to a CSV so we can manually eyeball data.
trainLabelsDF = withIndicators.iloc[:18000, 4]
trainLabelsDF.to_csv("./CSVs/trainLabelsDF.csv", index=False, header=False)
testFeaturesDF = withIndicators.iloc[18001:, 0:4]
testFeaturesDF.to_csv("./CSVs/testFeaturesDF.csv", index=False, header=False)
testLabelsDF = withIndicators.iloc[18001:, 4]
testLabelsDF.to_csv("./CSVs/testLabelsDF .csv", index=False, header=False)

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
def nb_num_features(train_features, train_labels, test_features): # Naive Bayers classifier.
    predictions = []
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    predictions = gnb.predict(test_features)
    return predictions

nb_predictions = nb_num_features(trainFeatures, trainLabels, testFeatures) # Predictions for the test set. 
print(f"1) Naive Bayes\n\t- Predicted class distribution:\t- {Counter(nb_predictions)}")
NB_error = 1 - accuracy_score(nb_predictions, testLabels)
NB_f1 = f1_score(nb_predictions, testLabels, average='macro')
print(f"\n---> NB\t\tError: {round(NB_error, 2)}\tMacro F1: {round(NB_f1, 2)}")


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
        for j in range(0, len(array)):\
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
