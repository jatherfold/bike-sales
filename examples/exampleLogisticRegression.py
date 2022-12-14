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
                             LogisticRegression(penalty = 'l1', solver = 'liblinear'))])
logisticParam = {'logisticRegression__C': np.logspace(-2, 2, 1000)}

    
logisticModel = Model(logisticRegPipe, 'L1 Logistic Regression')
logisticModel.optimiseHyperparameters(sandtonShopData.xTrainValid,
                                  sandtonShopData.yTrainValid,
                                  logisticParam, crossValObj)

#%% Train and Test Final Model

# treeModel = pickle.load(open('exampleTreeMdl.pkl', 'rb'))
logisticModel.train(sandtonShopData.xTrainValid, sandtonShopData.yTrainValid,
                sandtonShopData.processedFeatureNames)
logisticModel.test(sandtonShopData.xTest, sandtonShopData.yTest)
logisticModel.saveModel('l1LogisticMdlEurope.pkl')
logisticModel.printResults()
plt.figure()
plot_confusion_matrix(logisticModel.testConfusionMatrix, ['No', 'Yes'],
                      title = logisticModel.modelName, normalize=True)

logisticModel.plotFeatureImportance(12)


#%% Find known and unknown demographics

logisticModel.getDemographics(sandtonShopData)

#%% Answering the Question
