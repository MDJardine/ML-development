# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:01:43 2024

@author: user
"""

#### improvements to the model
## same start as basic model but if some chnages from looking at correlations 
## correlations have been done in R but will write Pthon script for them later 



#%%

# first read in the necessary packages
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#%%

## read in the data
## two sets:  trainign and test data

# training data
home_data = pd.read_csv('C:/Users/user/Documents/GitHub/ML-development/train.csv')

# and the test data 
test_data = pd.read_csv('C:/Users/user/Documents\\GitHub\\ML-development\\test.csv')

#%%

## have a look at the data 
list(test_data.columns)

#%%

test_data.describe()

#%%


## first we set the variable from the training data we want to predict
# call this y

y = home_data.SalePrice

#%%

## now select some features to train the model
features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

## now select the columsn from the training data that fit those features
# call this X
X = home_data[features]

