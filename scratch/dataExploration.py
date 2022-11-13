# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 08:31:14 2022

@author: john.atherfold
"""

#%% 0. Import the python libraries you think you'll require
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
import os
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import cross_val_score
from helpers.DataModule import plot_confusion_matrix, undummify

# Set-up Working Directory
abspath = os.path.abspath("dataExploration.py")
dname = os.path.dirname(abspath)
os.chdir(dname + "\\..")

from src.Model import Model

#%% 1. Load in the data.

bikeSaleData = pd.read_csv('./data/Bike_Buyer_Data_edited.txt')

# Data Cleaning
# At this point, the data looks a bit odd (not enough column names and lots of
# nans in final column)
# The cause of this is a data entry error in row 8. Most robust way to handle it is
# dropping rows and columns with nans
bikeSaleData = bikeSaleData.dropna(axis = 1, thresh = 900) # If more than 900 points are nan, drop the column
bikeSaleData = bikeSaleData.dropna(axis = 0) # Drop all rows with nans. Data: 1000 rows -> 996 rows. Still acceptable.


#%% 2. Explore the data.
#
# Visualise the data (How do the predictors/inputs relate to the responses/outputs?)

plt.figure()
plt.title('Purchased Bike')
bikeSaleData['Purchased Bike'].value_counts().plot(kind='bar')
# Response variable looks approximately balanced -> fine to train on without under/over sampling

# Visualise All Variables
# Using bar graph instead if histogram at the moment -> lots of categorical variables in data and I don't want to make any assumptions yet
allColumns = bikeSaleData.columns
for columnName in allColumns:
    plt.figure()
    plt.title(columnName)
    bikeSaleData[columnName].value_counts().plot(kind='bar')

# ID is a unique identifier -> Not useful for modelling, drop it.
bikeSaleData = bikeSaleData.drop('ID', 1)

# Cars feature has a "-1" entry -> Filter on positive or 0 cars only
bikeSaleData = bikeSaleData.loc[bikeSaleData['Cars'].astype(int) >= 0]

# Children feature - One entry has 12 children. This is either a data entry error
# or unlikely, and won't significantly affect the model -> filter on 5 or fewer children
bikeSaleData = bikeSaleData.loc[bikeSaleData['Children'].astype(int) <= 5]

# Other outliers are taken to be acceptable (Income, Age)

# Convert numeric features to numeric type. Most of them are saved as objects/strings
numericFeatureNames = ["Income", "Children", "Cars", "Age"]
for columnName in numericFeatureNames:
    bikeSaleData[columnName] = bikeSaleData[columnName].astype(float)

# Repeat similar plot as above, but colour by response -> See if there are any predictors that perfectly separate the classes
for columnName in bikeSaleData.columns[:-1]:
    bikeSaleData2 = bikeSaleData.groupby([columnName,"Purchased Bike"])[columnName].count().unstack('Purchased Bike').fillna(0)
    bikeSaleData2[['No','Yes']].plot(kind='bar', stacked=True, title = columnName, rot=90)

# There are no trivial solutions that perfectly separate the response -> simpler linear models are unlikely to be very successful

# Exploring the dependence in the numeric predictors
sns.pairplot(bikeSaleData)

# Numeric predictors appear to be linearly independent -> Good for the model; expecting each predictor to add unique information

# Some models require numeric data only -> perform one-hot encoding on Categorical Data to map categories to numbers
oneHotBikeSalesData = bikeSaleData.copy()
categoricalColumns = ["Marital Status", "Gender", "Education", "Occupation",
                      "Home Owner", "Commute Distance", "Region", "Purchased Bike"]
for columnName in categoricalColumns:
    oneHot = pd.get_dummies(oneHotBikeSalesData[columnName])
    oneHotBikeSalesData = oneHotBikeSalesData.drop(columnName, axis = 1)
    oneHotBikeSalesData = oneHotBikeSalesData.join(oneHot, rsuffix = columnName)

# Feature Clean-up - Some encoded features are redundant, i.e. "Gender" column
# is now "Male" and "Female" columns. Contains same information, so we only
# need one of the new columns.
columnsToDrop = ["Single", "Female", "No", "NoPurchased Bike"]
for columnName in columnsToDrop:
    oneHotBikeSalesData = oneHotBikeSalesData.drop(columnName, axis = 1)

oneHotBikeSalesData = oneHotBikeSalesData.rename(columns={'Yes':'Home Owner',
                                                         'YesPurchased Bike':'Purchased Bike'}) 

#%% 3. Formulate the Problem
# Problem Statement: Determine the type of customer that will buy a motorcycle
# In this case, model ressults are less important than the INSIGHT the models are able to provide

# Split the Test Set stratified on our target class. This can be checked by plotting
# histograms of yTest and yTrainValid and comparing them. They should look the same.
xTrainValid, xTest, yTrainValid, yTest = model_selection.train_test_split(
    oneHotBikeSalesData[oneHotBikeSalesData.columns[:-1]].values, oneHotBikeSalesData[oneHotBikeSalesData.columns[-1]].values,
    test_size = int(0.15*len(oneHotBikeSalesData)))


#%% 4. Create Baseline Model for Comparison
# Best Random guess
majorClass = max(bikeSaleData['Purchased Bike'].value_counts())
minorClass = min(bikeSaleData['Purchased Bike'].value_counts())
randomGuess = majorClass/(majorClass + minorClass)
print('-------------------------------------------------------------')
print('Random Guess: %6.3f' %randomGuess)
print('-------------------------------------------------------------')

# Starting with something super simple -> Default Logistic Regression (L2 regularisation)

logisticReg = LogisticRegression(solver = 'liblinear')
logisticReg.fit(xTrainValid, yTrainValid)
yTrainValidHat = logisticReg.predict(xTrainValid)
yTestHat = logisticReg.predict(xTest)
print('-------------------------------------------------------------')
print('Training Results (Default Logistic Regression)')
print('-------------------------------------------------------------')
print(confusion_matrix(yTrainValid, yTrainValidHat))
print("F1 Score: %6.4f" % f1_score(yTrainValid, yTrainValidHat))
print("Accuracy: %6.4f" % accuracy_score(yTrainValid, yTrainValidHat))
print('-------------------------------------------------------------')
print('-------------------------------------------------------------')
print('Testing Results (Default Logistic Regression)')
print('-------------------------------------------------------------')
print(confusion_matrix(yTest, yTestHat))
print("F1 Score: %6.4f" % f1_score(yTest, yTestHat))
print("Accuracy: %6.4f" % accuracy_score(yTest, yTestHat))
# The model has a Test accuracy of 61.7%. A random guess would have an accuracy of 51.9%.
# The F1 score is 0.6275 -> should be close to 1
# Model performs better than a random guess, but still not excellent

# Try something slightly more specialised -> Logistic Regression with default L1 regularisation
logisticReg = LogisticRegression(penalty = 'l1', solver = 'liblinear')
logisticReg.fit(xTrainValid, yTrainValid)
yTrainValidHat = logisticReg.predict(xTrainValid)
yTestHat = logisticReg.predict(xTest)
print('-------------------------------------------------------------')
print('')
print('-------------------------------------------------------------')
print('Training Results (L1 Logistic Regression)')
print('-------------------------------------------------------------')
print(confusion_matrix(yTrainValid, yTrainValidHat))
print("F1 Score: %6.4f" % f1_score(yTrainValid, yTrainValidHat))
print("Accuracy: %6.4f" % accuracy_score(yTrainValid, yTrainValidHat))
print('-------------------------------------------------------------')
print('-------------------------------------------------------------')
print('Testing Results (L1 Logistic Regression)')
print('-------------------------------------------------------------')
print(confusion_matrix(yTest, yTestHat))
print("F1 Score: %6.4f" % f1_score(yTest, yTestHat))
print("Accuracy: %6.4f" % accuracy_score(yTest, yTestHat))

# The model has a test accuracy of 65.1%. A random guess would have an accuracy of 51.9%.
# The F1 score is 0.6709 -> should be close to 1
# Improvement in the model. L1 norm is able to weigh some input features as 0. L2 norm is not.
featureWeights = pd.DataFrame(data = logisticReg.coef_, columns=oneHotBikeSalesData.columns[:-1]).transpose()
featureWeights.plot(kind = 'bar')

#%% 5. Identify a Suitable Machine Learning Model
# As expected from the Data analysis, default Linear models have limited success -> move to more complex models

crossValObj = KFold(n_splits=20)
# Fit the model and get results

# Logistic Regression - L1 Penalisation
logisticPipe = Pipeline([('scaler', preprocessing.MinMaxScaler()),
                         ('logisticRegression', LogisticRegression(penalty = 'l1', solver = 'liblinear'))])
logisticParam = {'logisticRegression__C': np.logspace(-2, 2, 1000)}

logisticModelL1 = Model(logisticPipe, 'L1 Logistic Regression')
logisticModelL1.optimiseHyperparameters(xTrainValid, yTrainValid, logisticParam, crossValObj)

# Logistic Regression - L2 Penalisation
logisticPipe = Pipeline([('scaler', preprocessing.MinMaxScaler()),
                         ('logisticRegression', LogisticRegression(penalty = 'l2', solver = 'liblinear'))])
logisticParam = {'logisticRegression__C': np.logspace(-2, 2, 1000)}

logisticModelL2 = Model(logisticPipe, 'L2 Logistic Regression')
logisticModelL2.optimiseHyperparameters(xTrainValid, yTrainValid, logisticParam, crossValObj)

# SVM - Best model according to k-fold cross-validation results
svcPipe = Pipeline([('scaler', preprocessing.MinMaxScaler()),
                    ('svc', SVC())])
svcParam = {
    'svc__C': Real(1e-4, 1e+0, prior='log-uniform'),
    'svc__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'svc__degree': Integer(1,4),
    'svc__kernel': Categorical(['linear', 'poly', 'rbf'])}
svcModel = Model(svcPipe, 'SVC')
svcModel.optimiseHyperparameters(xTrainValid, yTrainValid, svcParam, crossValObj)

# Decision Tree - Needs hyperparameter tuning, but most results aren't bad

treePipe = Pipeline([('scaler', preprocessing.MinMaxScaler()),
                     ('decisionTree', DecisionTreeClassifier())])
treeParam = {
    'decisionTree__criterion': Categorical(['gini', 'entropy', 'log_loss']),
    'decisionTree__max_depth': np.arange(1,900),
    'decisionTree__min_samples_split': np.arange(2,10),
    'decisionTree__min_samples_leaf': np.arange(1,10)}
    
treeModel = Model(treePipe, 'Decision Tree')
treeModel.optimiseHyperparameters(xTrainValid, yTrainValid, treeParam, crossValObj)


# print(metrics.confusion_matrix(yValidFull, yHatFull))
# print(metrics.f1_score(yValidFull, yHatFull))
# print(metrics.accuracy_score(yValidFull, yHatFull))
    

#%% 6. Compare Models
# Train final model on ALL training and validation data, then test on test data.

logisticModelL1.train(xTrainValid, yTrainValid, oneHotBikeSalesData.columns)
logisticModelL1.test(xTest, yTest)
logisticModelL1.saveModel('logisticMdlL1.pkl')
logisticModelL1.printResults()
plot_confusion_matrix(logisticModelL1.testConfusionMatrix, ['No', 'Yes'],
                      title = logisticModelL1.modelName, normalize=False)

logisticModelL2.train(xTrainValid, yTrainValid, oneHotBikeSalesData.columns)
logisticModelL2.test(xTest, yTest)
logisticModelL2.saveModel('logisticMdlL2.pkl')
logisticModelL2.printResults()
plot_confusion_matrix(logisticModelL2.testConfusionMatrix, ['No', 'Yes'],
                      title = logisticModelL2.modelName, normalize=False)

svcModel.train(xTrainValid, yTrainValid, oneHotBikeSalesData.columns)
svcModel.test(xTest, yTest)
svcModel.saveModel('svcMdl.pkl')
svcModel.printResults()
plot_confusion_matrix(svcModel.testConfusionMatrix, ['No', 'Yes'],
                      title = svcModel.modelName, normalize=False)

treeModel.train(xTrainValid, yTrainValid, oneHotBikeSalesData.columns)
treeModel.test(xTest, yTest)
treeModel.saveModel('treeMdl.pkl')
treeModel.printResults()
plot_confusion_matrix(treeModel.testConfusionMatrix, ['No', 'Yes'],
                      title = treeModel.modelName, normalize=False)

#%% 7. Answer the Question
# Consider feature importances of each of the models to determine the customer base

logisticModelL1.plotFeatureImportance(12)
logisticModelL2.plotFeatureImportance(12)
svcModel.plotFeatureImportance(12, xTrainValid, yTrainValid)
treeModel.plotFeatureImportance(12)

#%% Demographic Analysis

# Consider the full dataset - what are the blindspots of the model?
#   Consider Decision Tree only -> best performance on Test set

xFull = np.concatenate((xTrainValid, xTest))
yFull = np.concatenate((yTrainValid, yTest))
shuffledBikeSalesData = pd.DataFrame(data = np.concatenate((xTest, yTest[:, np.newaxis]), axis = 1),
                                     columns = oneHotBikeSalesData.columns)

# Close look at true positives. Plot Distributions.
truePositiveIdx =  np.logical_and(treeModel.yTestHat == 1, yTest == 1)
knownDemographic = shuffledBikeSalesData.loc[truePositiveIdx]

# Close look at false negatives. Plot Distributions.
falseNegativeIdx = np.logical_and(treeModel.yTestHat == 0, yTest == 1)
unknownDemographic = shuffledBikeSalesData.loc[falseNegativeIdx]

# Plots
importantFeatures = ['Age', 'Income', 'Married', 'Cars', 'Male']
feature = 'Income'
plt.figure()
plt.bar(knownDemographic[feature].value_counts().sort_index().index.astype(str), knownDemographic[feature].value_counts().sort_index().values, label = 'Known Demographic', alpha = 0.5)
plt.bar(unknownDemographic[feature].value_counts().sort_index().index.astype(str), unknownDemographic[feature].value_counts().sort_index().values, label = 'Unknown Demographic', alpha = 0.5)
plt.title(feature)
plt.xticks(rotation = 45)
plt.legend()

for feature in importantFeatures:
    plt.figure()
    plt.bar(knownDemographic[feature].value_counts().index, knownDemographic[feature].value_counts().values, label = 'Known Demographic', alpha = 0.5)
    plt.bar(unknownDemographic[feature].value_counts().index, unknownDemographic[feature].value_counts().values, label = 'Unknown Demographic', alpha = 0.5)
    plt.title(feature)
    plt.legend()