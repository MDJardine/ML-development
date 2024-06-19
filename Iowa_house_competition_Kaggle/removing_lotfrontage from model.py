# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:12:29 2024

@author: user
"""

## previously I've had some probelms with lot frontage which I thought was a good feature to include 
## however replacign it with zeros or remoing the rows wasn't great
## what abut just removing it?


#%%

# first read in the necessary packages
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#%%

## read in the data

## two sets:  trainign and test data

#
# training data
home_data = pd.read_csv('C:/Users/user/Documents/GitHub/ML-development/Iowa_house_competition_Kaggle/train.csv')

# and the test data 
test_data = pd.read_csv('C:/Users/user/Documents/GitHub/ML-development/Iowa_house_competition_Kaggle\\test.csv')

#%%

## have a look at the data 
list(home_data.columns)

#%%

## have a look at the the data too
home_data.describe()

#%%

## choosing features from analysis done in R
features2 = ["OverallQual", "YearRemodAdd", "MasVnrArea", "GrLivArea", "FullBath", "TotRmsAbvGrd", 
            "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "SalePrice"]

## now select the columsn from the training data that fit those features
# call this X
X2 = home_data[features2]

## have a look to check this
X2.describe()

#%%

## check for null values
X2.isnull().sum()
## so there are 8 rows which have an NA for the MasVnrArea


#%%

# remove the 8 rows of NANs
X2 = X2.dropna(axis="rows")
X2.shape[0]
## so we still have 1452 rows of data - more is data is better usually?


#%%

## defien the target variable y 
y2 = X2.SalePrice

#%%
# and correspondingly drop the variable from the X variable set
X2_E = X2.drop(['SalePrice'], axis=1)


## now one way to do this is to split the home_data into a traing and test set
## not really needed here but done for properity at this time
train_X2, val_X2, train_y2, val_y2 = train_test_split(X2_E, y2, random_state=1)

#%%

## now build a random forest model using these four sets 
rf_model_E = RandomForestRegressor(random_state=5)

## fit model to data
rf_model_E.fit(train_X2, train_y2)

## predict values based on the test set
rf_val_prediction_E = rf_model_E.predict(val_X2)

## calculate the error for sales price
rf_val_mae_E = mean_absolute_error(rf_val_prediction_E, val_y2)

#%%

## ok so now we find what the error in the orediction actually are
print("Validation MAE for Random Forest Model {:,.0f}".format(rf_val_mae_E))

## ok so it is marginally better
# $22,050 - so better to have lot frontage in 

## next i wnat to include what in the dta aare binary varibales 
# noted donwn a few good ones from my R analayis 
# would alos be good to repate that r analyis in python
