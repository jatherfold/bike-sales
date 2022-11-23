# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:41:05 2022

@author: john.atherfold
"""

# Detailed Data Analysis Script

#%% Import libraries 
import sys
import matplotlib.pyplot as plt

# adding src to the system path
sys.path.insert(0, './src')

from src.Data import Data
import pandas as pd

#%% Create Data Object

sandtonShopData = Data('./data/Bike_Buyer_Data_edited.txt') #For example - may be location specific

sandtonShopData.loadData()

sandtonShopData.preprocessData(filterOn = None)

purchasedBikeIdx = sandtonShopData.processedData['Purchased Bike'] == 'Yes'

#%%
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
plt.title('Bikes Sold per Region, Compared to Children')

#%%

pd.crosstab(sandtonShopData.processedData['Region'][purchasedBikeIdx],sandtonShopData.processedData['Commute Distance'][purchasedBikeIdx]).transpose().loc[['0-1 Miles', '1-2 Miles', '2-5 Miles', '5-10 Miles', '10+ Miles']].plot.bar(rot = 15)
plt.title('Bikes Sold per Region, Compared to Commute Distance')

#%%
pd.crosstab(sandtonShopData.processedData['Region'][purchasedBikeIdx],sandtonShopData.processedData['Age'][purchasedBikeIdx]).transpose().plot.bar(rot = 15)
plt.title('Bikes Sold per Region, Compared to Age')

#%%
pd.crosstab(sandtonShopData.processedData['Region'][purchasedBikeIdx],sandtonShopData.processedData['Income'][purchasedBikeIdx]).transpose().plot.bar(rot = 15)
plt.title('Bikes Sold per Region, Compared to Income')

#%%
pd.crosstab(sandtonShopData.processedData['Region'][purchasedBikeIdx],sandtonShopData.processedData['Occupation'][purchasedBikeIdx]).transpose().plot.bar(rot = 15)
plt.title('Bikes Sold per Region, Compared to Profession')