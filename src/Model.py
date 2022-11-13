# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:08:46 2022

@author: john.atherfold
"""

import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pickle

class Model():
    def __init__(self, pipeline, modelName):
        self.pipeline = pipeline
        self.modelName = modelName
        
    def optimiseHyperparameters(self, xTrainValid, yTrainValid, param, cvObj, numIter = 100):
        self.crossValObject = cvObj
        randomSearch = BayesSearchCV(estimator = self.pipeline, search_spaces = param,
                                      n_iter = numIter, cv = cvObj, verbose = 5,
                                      n_jobs = -1, scoring = 'f1')
        randomSearch.fit(xTrainValid, yTrainValid)
        self.bestMdl = randomSearch.best_estimator_
        self.crossValScores = cross_val_score(self.bestMdl, xTrainValid, yTrainValid, cv = cvObj)

        print('')
        print('')
        print('-------------------------------------------------------------')
        print('Cross-Validation Results (Optimised ' + self.modelName + ')')
        print('-------------------------------------------------------------')
        print('Median F1 Score: %6.4f' %np.median(self.crossValScores))
    
    def train(self, xTrainValid, yTrainValid, featureNames):
        self.featureNames = featureNames
        
        self.bestMdl.fit(xTrainValid, yTrainValid)
        self.yTrainValidHat = self.bestMdl.predict(xTrainValid)
        self.residuals = yTrainValid - self.yTrainValidHat
        
        self.trainConfusionMatrix = confusion_matrix(yTrainValid, self.yTrainValidHat)
        self.trainF1Score = f1_score(yTrainValid, self.yTrainValidHat)
        self.trainAccuracy = accuracy_score(yTrainValid, self.yTrainValidHat)
        
    def test(self, xTest, yTest):
        self.yTestHat = self.bestMdl.predict(xTest)
        self.testErrors = yTest - self.yTestHat
        
        self.testConfusionMatrix = confusion_matrix(yTest, self.yTestHat)
        self.testF1Score = f1_score(yTest, self.yTestHat)
        self.testAccuracy = accuracy_score(yTest, self.yTestHat)
        
    def printResults(self):
        print('')
        print('')
        print('')
        print('-------------------------------------------------------------')
        print('Training Results (' + self.modelName + ')')
        print('-------------------------------------------------------------')
        print(self.trainConfusionMatrix)
        print("F1 Score: %6.4f" % self.trainF1Score)
        print("Accuracy: %6.4f" % self.trainAccuracy)
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
        print('Testing Results (' + self.modelName + ')')
        print('-------------------------------------------------------------')
        print(self.testConfusionMatrix)
        print("F1 Score: %6.4f" % self.testF1Score)
        print("Accuracy: %6.4f" % self.testAccuracy)
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
        
    def plotCrossValScores(self):
        crossValDF = pd.DataFrame(data = self.crossValScores).transpose()
        fig, ax = plt.subplots()
        crossValDF.plot(ax = ax)
        ax.hline(y = np.median(self.crossValScores), linewidth = 2, color = 'r')
        
    def plotFeatureImportance(self, topN, xTrainValid = None, yTrainValid = None):
        if 'Logistic Regression' in self.modelName:
            featureWeightData = self.bestMdl[1].coef_
            sortedIdx = abs(featureWeightData).argsort()
            plt.figure()
            plt.barh(self.featureNames[sortedIdx][0][-topN:], featureWeightData[0][sortedIdx][0][-topN:])
            plt.xlabel(self.modelName + " Feature Importance")
        elif 'SVC' in self.modelName:
            permImportances = permutation_importance(self.bestMdl, xTrainValid, yTrainValid)
            featureWeightData = permImportances.importances_mean
            sortedIdx = abs(featureWeightData).argsort()
            plt.figure()
            plt.barh(self.featureNames[sortedIdx][-topN:], featureWeightData[sortedIdx][-topN:])
            plt.xlabel(self.modelName + " Feature Importance")
        elif 'Decision Tree' in self.modelName:
            featureWeightData = self.bestMdl[1].feature_importances_
            sortedIdx = abs(featureWeightData).argsort()
            plt.figure()
            plt.barh(self.featureNames[sortedIdx][-topN:], featureWeightData[sortedIdx][-topN:])
            plt.xlabel(self.modelName + " Feature Importance")
        plt.subplots_adjust(left = 0.4)
        
    def getDemographics(self, shopData):
        fullTestData = np.concatenate((shopData.xTest, shopData.yTest[:, np.newaxis]), axis = 1)
        shuffledBikeSalesData = pd.DataFrame(data = fullTestData,
                                             columns = shopData.processedFeatureNames)
        # Close look at true positives. Plot Distributions.
        truePositiveIdx =  np.logical_and(self.yTestHat == 1, shopData.yTest == 1)
        self.knownDemographic = shuffledBikeSalesData.loc[truePositiveIdx]

        # Close look at false negatives. Plot Distributions.
        falseNegativeIdx = np.logical_and(self.yTestHat == 0, shopData.yTest == 1)
        self.unknownDemographic = shuffledBikeSalesData.loc[falseNegativeIdx]
    
    def saveModel(self, fileName):
        pickle.dump(self, open(fileName, 'wb'))