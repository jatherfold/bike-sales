# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:33:04 2022

@author: john.atherfold
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data():
    def __init__(self, dataFile):
        self.fileName = dataFile
        
    def loadData(self):
        self.rawData = pd.read_csv(self.fileName)
        self.rawData = self.rawData.dropna(axis = 1, thresh = 900) # If more than 900 points are nan, drop the column
        self.rawData = self.rawData.dropna(axis = 0) # Drop all rows with nans. Data: 1000 rows -> 996 rows. Still acceptable.

    def preprocessData(self):
        self.processedData = self.rawData.copy()
        # ID is a unique identifier -> Not useful for modelling, drop it.
        self.processedData = self.processedData.drop('ID', 1)

        # Cars feature has a "-1" entry -> Filter on positive or 0 cars only
        self.processedData = self.processedData.loc[self.processedData['Cars'].astype(int) >= 0]

        # Children feature - One entry has 12 children. This is either a data entry error
        # or unlikely, and won't significantly affect the model -> filter on 5 or fewer children
        self.processedData = self.processedData.loc[self.processedData['Children'].astype(int) <= 5]

        # Other outliers are taken to be acceptable (Income, Age)

        # Convert numeric features to numeric type. Most of them are saved as objects/strings
        numericFeatureNames = ["Income", "Children", "Cars", "Age"]
        for columnName in numericFeatureNames:
            self.processedData[columnName] = self.processedData[columnName].astype(float)
        
        self.oneHotData = self.processedData.copy()
        categoricalColumns = ["Marital Status", "Gender", "Education", "Occupation",
                              "Home Owner", "Commute Distance", "Region", "Purchased Bike"]
        for columnName in categoricalColumns:
            oneHot = pd.get_dummies(self.oneHotData[columnName])
            self.oneHotData = self.oneHotData.drop(columnName, axis = 1)
            self.oneHotData = self.oneHotData.join(oneHot, rsuffix = columnName)

        # Feature Clean-up - Some encoded features are redundant, i.e. "Gender" column
        # is now "Male" and "Female" columns. Contains same information, so we only
        # need one of the new columns.
        columnsToDrop = ["Single", "Female", "No", "NoPurchased Bike"]
        for columnName in columnsToDrop:
            self.oneHotData = self.oneHotData.drop(columnName, axis = 1)

        self.oneHotData = self.oneHotData.rename(columns={'Yes':'Home Owner',
                                                          'YesPurchased Bike':'Purchased Bike'})
        self.processedFeatureNames = self.oneHotData.columns
        
    def trainTestSplit(self, testFrac):
        self.xTrainValid, self.xTest, self.yTrainValid, self.yTest = train_test_split(
            self.oneHotData[self.oneHotData.columns[:-1]].values,
            self.oneHotData[self.oneHotData.columns[-1]].values,
            test_size = int(testFrac*len(self.oneHotData)))