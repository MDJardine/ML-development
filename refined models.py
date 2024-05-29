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

## choosing features from analysis done in R
features_R = ["LotFrontage", "OverallQual", "YearRemodAdd", "MasVnrArea", "GrLivArea", "FullBath", "TotRmsAbvGrd", 
            "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF"  ]

## now select the columsn from the training data that fit those features
# call this X
X = home_data[features_R]


#%%

## now one way to do this is to split the home_data into a traing and test set
## not really needed here but done for properity at this time
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#%%

## now build a random forest model using these four sets 
rf_model_R = RandomForestRegressor(random_state=1)
rf_model_R.fit(train_X, train_y)
rf_val_pred_R = rf_model_R.predict(val_X)
rf_val_mae_R = mean_absolute_error(rf_val_pred_R, val_y)

## error with NAn values
## so do we just remove them or haow can be get rnadom forest to deal with them?

#%%

## ok so now we find what the erros in the orediction actually are
print("Validation MAE for Random Forest Model {:,.0f}".format(rf_val_mae_R))




