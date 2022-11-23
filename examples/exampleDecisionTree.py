# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:29:39 2022

@author: john.atherfold
"""

#%% Import libraries 
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
import os
from skopt.space import Real, Categorical, Integer
import sys
import matplotlib.pyplot as plt

# adding src to the system path
sys.path.insert(0, './src')

from src.Model import Model
from src.Data import Data
from helpers.DataModule import plot_confusion_matrix
import pickle
import pandas as pd

#%% Create Data Object

sandtonShopData = Data('./data/Bike_Buyer_Data_edited.txt') #For example - may be location specific

sandtonShopData.loadData()

sandtonShopData.preprocessData(filterOn = None)

#%% Split Data for ML Workflow

sandtonShopData.trainTestSplit('OneHot', 0.15)

#%% Setup for and ML Workflow

crossValObj = KFold(n_splits=20)
treePipe = Pipeline([#('scaler', preprocessing.MinMaxScaler()),
                     ('decisionTree', DecisionTreeClassifier())])
treeParam = {
    'decisionTree__criterion': Categorical(['gini', 'entropy', 'log_loss']),
    'decisionTree__max_depth': np.arange(1,100),
    'decisionTree__min_samples_split': np.arange(2,100),
    'decisionTree__min_samples_leaf': np.arange(1,100)}
    
treeModel = Model(treePipe, 'Decision Tree')
treeModel.optimiseHyperparameters(sandtonShopData.xTrainValid,
                                  sandtonShopData.yTrainValid,
                                  treeParam, crossValObj)

#%% Train and Test Final Model

# treeModel = pickle.load(open('exampleTreeMdl.pkl', 'rb'))
treeModel.train(sandtonShopData.xTrainValid, sandtonShopData.yTrainValid,
                sandtonShopData.processedFeatureNames)
treeModel.test(sandtonShopData.xTest, sandtonShopData.yTest)
treeModel.saveModel('treeMdlEurope.pkl')
treeModel.printResults()
plot_confusion_matrix(treeModel.testConfusionMatrix, ['No', 'Yes'],
                      title = treeModel.modelName, normalize=False)

treeModel.plotFeatureImportance(12)
plt.figure()
plot_tree(treeModel.bestMdl.named_steps['decisionTree'],
          feature_names = treeModel.featureNames,
          class_names = ['True', 'False'], filled = True,
          fontsize=9, max_depth = 4)

#%% Find known and unknown demographics

treeModel.getDemographics(sandtonShopData)

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
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Region', 'Income']).count()['Purchased Bike'].plot.bar()
#%%

pd.crosstab(sandtonShopData.processedData['Region'][purchasedBikeIdx],sandtonShopData.processedData['Cars'][purchasedBikeIdx]).transpose().plot.bar()
plt.title('Bikes Sold per Region, Compared to Cars Owned')

#%%

pd.crosstab(sandtonShopData.processedData['Region'][purchasedBikeIdx],sandtonShopData.processedData['Children'][purchasedBikeIdx]).transpose().plot.bar()
plt.title('Bikes Sold per Region, Compared to Cars Owned')

#%%

pd.crosstab(sandtonShopData.processedData['Region'][purchasedBikeIdx],sandtonShopData.processedData['Cars'][purchasedBikeIdx]).transpose().plot.bar()
plt.title('Bikes Sold per Region, Compared to Cars Owned')
sandtonShopData.processedData[purchasedBikeIdx].groupby(['Region', 'Commute Distance']).count()['Purchased Bike'].plot(kind='bar', rot = 15)

sandtonShopData.processedData[purchasedBikeIdx].groupby(['Region', 'Age']).count()['Purchased Bike'].plot(kind='bar', rot = 15)
