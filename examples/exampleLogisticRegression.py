# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:16:04 2022

@author: john.atherfold
"""

#%% Import libraries 
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
import matplotlib.pyplot as plt

# adding src to the system path
sys.path.insert(0, './src')

from src.Model import Model
from src.Data import Data
from helpers.DataModule import plot_confusion_matrix
import pickle

#%% Create Data Object

sandtonShopData = Data('./data/Bike_Buyer_Data_edited.txt') #For example - may be location specific

sandtonShopData.loadData()

sandtonShopData.preprocessData(filterOn = ['Region', 'Europe'])

#%% Split Data for ML Workflow

sandtonShopData.trainTestSplit('OneHot', 0.15)

#%% Setup for and ML Workflow

crossValObj = KFold(n_splits=20)
logisticRegPipe = Pipeline([('scaler', preprocessing.MinMaxScaler()),
                            ('logisticRegression',
                             LogisticRegression(solver = 'liblinear'))])
logisticParam = {'logisticRegression__C': np.logspace(-2, 2, 1000)}

    
logisticModel = Model(logisticRegPipe, 'Logistic Regression')
logisticModel.optimiseHyperparameters(sandtonShopData.xTrainValid,
                                  sandtonShopData.yTrainValid,
                                  logisticParam, crossValObj)

#%% Train and Test Final Model

# treeModel = pickle.load(open('exampleTreeMdl.pkl', 'rb'))
logisticModel.train(sandtonShopData.xTrainValid, sandtonShopData.yTrainValid,
                sandtonShopData.processedFeatureNames)
logisticModel.test(sandtonShopData.xTest, sandtonShopData.yTest)
logisticModel.saveModel('logisticMdlEurope.pkl')
logisticModel.printResults()
plot_confusion_matrix(logisticModel.testConfusionMatrix, ['No', 'Yes'],
                      title = logisticModel.modelName, normalize=False)

logisticModel.plotFeatureImportance(12)


#%% Find known and unknown demographics

logisticModel.getDemographics(sandtonShopData)

#%% Answering the Question

purchasedBikeIdx = sandtonShopData.processedData['Purchased Bike'] == 'Yes'
# Start by taking a look at customer segmentation by the most important features
plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Age']).count()['Purchased Bike'].plot(kind='bar')

# Mainly between 26 - 68. Quite a substantial range. Dig a bit deeper.
plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Cars', 'Age']).count()['Purchased Bike'].plot(kind='bar')
# 0 Cars: Between 32 and 51
# 1 Car: Between 26 and 59
# 2 Cars: Between 26 and 35, and between 40 and 69
# 3 Cars: Between 40 and 60
# 4 Cars: Between 40 and 50

# Continuing to dig deeper...
plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Income']).count()['Purchased Bike'].plot(kind='bar')
# Lower to Middle income earners more likely to buy cars
plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Cars', 'Income']).count()['Purchased Bike'].plot(kind='bar')

plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Children']).count()['Purchased Bike'].plot(kind='bar')

# Children is quite deciding. Fewer children -> more likely to purchase a bike
plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Children', 'Income']).count()['Purchased Bike'].plot(kind='bar')

plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Gender', 'Children']).count()['Purchased Bike'].plot(kind='bar')

plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Occupation', 'Income']).count()['Purchased Bike'].plot(kind='bar', rot = 15)

plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Commute Distance', 'Income']).count()['Purchased Bike'].plot(kind='bar', rot = 15)

plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Occupation', 'Commute Distance']).count()['Purchased Bike'].plot(kind='bar', rot = 15)

plt.figure()
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Home Owner', 'Age']).count()['Purchased Bike'].plot(kind='bar', rot = 15)

plt.figure() # This is a good one. Interesting to see the differences.
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Region', 'Income']).count()['Purchased Bike'].plot(kind='bar', rot = 15)

sandtonShopData.processedData[purchasedBikeIdx].groupby(['Region', 'Cars']).count()['Purchased Bike'].plot(kind='bar', rot = 15)

sandtonShopData.processedData[purchasedBikeIdx].groupby(['Region', 'Commute Distance']).count()['Purchased Bike'].plot(kind='bar', rot = 15)

sandtonShopData.processedData[purchasedBikeIdx].groupby(['Region', 'Age']).count()['Purchased Bike'].plot(kind='bar', rot = 15)
