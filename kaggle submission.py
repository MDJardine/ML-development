# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:12:29 2024

@author: user
"""
############################################### 
#### kaggle submission script ####


### this will just be a copy of the best model I've built so far 


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
y
#%%

## choosing features from analysis done in R
features_R = ["LotFrontage", "OverallQual", "YearRemodAdd", "MasVnrArea", "GrLivArea", "FullBath", "TotRmsAbvGrd", 
            "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF"  ]

## now select the columsn from the training data that fit those features
# call this X
X = home_data[features_R]

#%%

## check for null values
X.isnull().sum()
## the vast majority are in the Lot frontage - 259 NAns
# makes sense if there is no front lot
# although also 8 in MasVnr area

#%%

## chnage Nans to 0

# create new dataframe name to avoid confusion
## and replace the Nans with 0
X_rep0 = X.fillna(0)

## now we build the model
#%%

## now one way to do this is to split the home_data into a traing and test set
## not really needed here but done for properity at this time
train_X_rep0, val_X_rep0, train_y, val_y = train_test_split(X_rep0, y, random_state=1)

#%%

## now build a random forest model using these four sets 
rf_model_R_rep0 = RandomForestRegressor(random_state=5)

## fit model to data
rf_model_R_rep0.fit(train_X_rep0, train_y)

## predict values for test set
rf_val_pred_R_rep0 = rf_model_R_rep0.predict(val_X_rep0)

## calculate the error for sales price
rf_val_mae_R_rep0 = mean_absolute_error(rf_val_pred_R_rep0, val_y)

#%%

## ok so now we find what the error in the orediction actually are
print("Validation MAE for Random Forest Model {:,.0f}".format(rf_val_mae_R_rep0))

## ok so it is marginally better
# $20,257
