import numpy as np
import pandas as pd
import talib, subprocess, platform, os, math, copy
from collections import Counter
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

RSI_Period = 14
Fast_Period = 12 # MACD fast period initialised at 12, subject to change.
Slow_Period = 26 # MACD slow period initialised at 26, subject to change.   
Signal_Period = 9 # MACD signal period initialised at 9, subject to change.

def profitLoss(testCloses, predictions):
    profit_loss = 0
    for i in range(len(predictions)-1):
        if predictions[i] == 1:
            profit_loss = profit_loss + (testCloses[i+1]-testCloses[i])
    print(f"Outcome is: ${round(profit_loss, 2)}")
    return profit_loss

def getFeaturesAndLabels(justLows, justHighs, justCloses, justVolume):
    justLows = np.array(justLows)
    justHighs = np.array(justHighs)
    justCloses = np.array(justCloses)
    justVolume = np.array(justVolume)
    rsi = talib.RSI(justCloses, RSI_Period)
    macd, macdsignal, macdhist = talib.MACD(justCloses, Fast_Period, Slow_Period, Signal_Period) # MACD calculated using candle close data, Fast/Slow/Signal Periods can be changed at initialisation point.
    upperband, middleband, lowerband = talib.BBANDS(justCloses, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bbw = ((upperband - lowerband) / middleband)
    real = talib.OBV(justCloses, justVolume)
    slowk, slowd = talib.STOCH(justHighs, justLows, justCloses, fastk_period=14, slowk_period=1, slowk_matype=0, slowd_period=3, slowd_matype=0)

    latestData = {'close': justCloses, 'RSI': rsi, 'MACD': macd, 'MACDSIGNAL': macdsignal,'MACDHIST': macdhist, 'BBW': bbw, 'OBV': real, 'SLOWK': slowk, 'SLOWD': slowd}
    _df_ = pd.DataFrame(latestData)
    _df_["Label"] = 0 
    _df_["Label"] = np.where((_df_['close']) + 15 < _df_['close'].shift(-1), 1, 0)
    _df_.to_csv("./CSVs/withFeatures.csv", index=True, header=True)
    
    _df_.drop(    # Drop the columns that aren't good features.
        labels = ["close", "SLOWK", "SLOWD"],
        axis = 1,
        inplace = True
        )
    _df_.drop(    # Drop the first 33 rows as they contain blank cells.
            labels = range(0, 33),
            axis=0,
            inplace = True
            )

    scaler = MinMaxScaler()
    _df_[["BBW", "OBV"]] = scaler.fit_transform(_df_[["BBW", "OBV"]])
    numberOfFeatures = _df_.shape[1]
    numberOfCandles = _df_.shape[0]
    print(f"There are {numberOfFeatures} columns, and {numberOfCandles} rows.")

    """
    Below we partition the whole dataset into 4 parts: training data, training labels, test data, test labels.
    It's 80/10 split -- 80% of candles are used for training, and the other 80% is the test set where make the
    predictions and then compare them to the actual test labels.
    """
    trainFeaturesDF = _df_.iloc[:17435, 0:numberOfFeatures-1]
    trainFeaturesDF.to_csv("./CSVs/trainFeatures.csv", index=False, header=False) # Save to a CSV so we can manually eyeball data.
    trainLabelsDF = _df_.iloc[:17435, numberOfFeatures-1]
    trainLabelsDF.to_csv("./CSVs/trainLabels.csv", index=False, header=False)
    testFeaturesDF = _df_.iloc[17436:, 0:numberOfFeatures-1]
    testFeaturesDF.to_csv("./CSVs/testFeatures.csv", index=False, header=False)
    testLabelsDF = _df_.iloc[17436:, numberOfFeatures-1]
    testLabelsDF.to_csv("./CSVs/testLabels.csv", index=False, header=False)
    _df_.to_csv("./CSVs/afterModifications.csv", index=True, header=True) # The newly created CSV file (made on line 23) is overwritten with the features and labels inserted.

    featureNames = list(trainFeaturesDF.columns.values) 
    print(f"The features currently selected for training are: {featureNames}")
    trainFeatures = trainFeaturesDF.to_numpy() # Convert the partitioned dataframes above into numpy arrays (needed for the classifiers).
    trainLabels = trainLabelsDF.to_numpy()
    testLabels = testLabelsDF.to_numpy()
    testFeatures = testFeaturesDF.to_numpy()
    print("Dimensions of the partitioned dataframes:\n\t- trainFeatures: {}\n\t- trainLabels: {}\n\t- testFeatures: {}\n\t- testLabels: {}".format(trainFeatures.shape, trainLabels.shape, testFeatures.shape, testLabels.shape))
    # mutualInformation(trainFeatures, trainLabels, featureNames)

    print("\n------\nData is now cleaned up and partioned with MI calculated, so let's apply the classifiers.\n------\n")
    return trainFeatures, trainLabels, testFeatures, testLabels

def NB_Classifier(train_features, train_labels, test_features, test_labels): # Naive Bayers classifier.
    predictions = []
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    predictions = gnb.predict(test_features)
    print(f"Actual test labels class distribution:\t- {Counter(test_labels)}")
    print(f"1) Naive Bayes\n\t- Predicted class distribution:\t- {Counter(predictions)}")
    NB_error = 1 - accuracy_score(predictions, test_labels)
    NB_f1 = f1_score(predictions, test_labels, average='macro')
    print(f"\n---> NB\t\tError: {round(NB_error, 2)}\tMacro F1: {round(NB_f1, 2)}")
    return predictions

def normalizer(array, mean, std):
    normalized_array = copy.deepcopy(array)
    for i in range(0, len(array[0])):
        for j in range(0, len(array)):
            normalized_array[j][i] = (array[j][i] - mean[i])/std[i]  
    return normalized_array

def LR_Classifier(train_features, train_labels, test_features, test_labels):
    lr_predictions = []
    trainFeaturesNormalised = []
    testFeaturesNormalised = []

    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    np.set_printoptions(suppress=True)

    trainFeaturesNormalised = normalizer(train_features, mean, std)
    testFeaturesNormalised = normalizer(test_features, mean, std)

    logisticRegr = LogisticRegression()
    logisticRegr.fit(trainFeaturesNormalised, train_labels)
    lr_predictions = logisticRegr.predict(testFeaturesNormalised)

    score = logisticRegr.score(testFeaturesNormalised, test_labels)
    print(score)
    lr_err = 1 - accuracy_score(lr_predictions, test_labels)
    lr_f1 = f1_score(lr_predictions, test_labels, average='macro')

    print(f"\n2) Logistic Regression\n\t- Predicted class distribution:\t{Counter(lr_predictions)}")
    print(f'\t- The coefficients for this LR classifier are:\t{logisticRegr.coef_}')
    print(f"\n---> LR\t\tError: {round(lr_err, 2)}\tMacro F1: {round(lr_f1, 2)}")

def mutualInformation(trainFeatures, trainLabels, featureNames):
    """
    Mutual Information (MI) measures the correlation of each feature with the labels; the degree as to which
    each feature affects the label. Optional metric, but I included it for fun.
    """
    highest_mi_feature_name = ""                # feature with highest MI
    lowest_mi_feature_name = ""                 # feature with lowest MI
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
Current problem: Both classifiers are predicting 0 as the label for every candle. 
83% is the given accuracy only because the test labels are 83% 0. Thus, it's a misleading accuracy.
"""
