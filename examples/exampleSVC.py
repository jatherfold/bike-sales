# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:31:56 2022

@author: john.atherfold
"""

#%% Import libraries 
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import sys
import matplotlib.pyplot as plt

# adding src to the system path
sys.path.insert(0, './src')

from skopt.space import Real, Categorical, Integer
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
svcPipe = Pipeline([('scaler', preprocessing.MinMaxScaler()),
                    ('svc', SVC())])
svcParam = {
    'svc__C': Real(1e-4, 1e+0, prior='log-uniform'),
    'svc__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'svc__degree': Integer(1,4),
    'svc__kernel': Categorical(['linear', 'poly', 'rbf'])}
svcModel = Model(svcPipe, 'SVC')
svcModel.optimiseHyperparameters(sandtonShopData.xTrainValid,
                                  sandtonShopData.yTrainValid,
                                  svcParam, crossValObj)

#%% Train and Test Final Model

# treeModel = pickle.load(open('exampleTreeMdl.pkl', 'rb'))
svcModel.train(sandtonShopData.xTrainValid, sandtonShopData.yTrainValid,
                sandtonShopData.processedFeatureNames)
svcModel.test(sandtonShopData.xTest, sandtonShopData.yTest)
svcModel.saveModel('l1LogisticMdlEurope.pkl')
svcModel.printResults()
plt.figure()
plot_confusion_matrix(svcModel.testConfusionMatrix, ['No', 'Yes'],
                      title = svcModel.modelName, normalize=True)

svcModel.plotFeatureImportance(12)


#%% Find known and unknown demographics

svcModel.getDemographics(sandtonShopData)

#%% Answering the Question