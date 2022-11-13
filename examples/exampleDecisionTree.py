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

# Set-up Working Directory
abspath = os.path.abspath("dataExploration.py")
dname = os.path.dirname(abspath)
os.chdir(dname + "\\..")

from src.Model import Model
from src.Data import Data
from helpers.DataModule import plot_confusion_matrix

#%% Create Data Object

sandtonShopData = Data('./data/Bike_Buyer_Data_edited.txt') #For example - may be location specific

sandtonShopData.loadData()

sandtonShopData.preprocessData()

#%% Split Data for ML Workflow

sandtonShopData.trainTestSplit(0.15)

#%% Setup for and ML Workflow

crossValObj = KFold(n_splits=20)
treePipe = Pipeline([('scaler', preprocessing.MinMaxScaler()),
                     ('decisionTree', DecisionTreeClassifier())])
treeParam = {
    'decisionTree__criterion': Categorical(['gini', 'entropy', 'log_loss']),
    'decisionTree__max_depth': np.arange(1,900),
    'decisionTree__min_samples_split': np.arange(2,10),
    'decisionTree__min_samples_leaf': np.arange(1,10)}
    
treeModel = Model(treePipe, 'Decision Tree')
treeModel.optimiseHyperparameters(sandtonShopData.xTrainValid,
                                  sandtonShopData.yTrainValid,
                                  treeParam, crossValObj)

#%% Train and Test Final Model

treeModel.train(sandtonShopData.xTrainValid, sandtonShopData.yTrainValid,
                sandtonShopData.processedFeatureNames)
treeModel.test(sandtonShopData.xTest, sandtonShopData.yTest)
treeModel.saveModel('exampleTreeMdl.pkl')
treeModel.printResults()
plot_confusion_matrix(treeModel.testConfusionMatrix, ['No', 'Yes'],
                      title = treeModel.modelName, normalize=False)

treeModel.plotFeatureImportance(12)

#%% Find known and unknown demographics

treeModel.getDemographics(sandtonShopData)