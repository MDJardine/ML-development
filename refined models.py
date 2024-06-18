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

## two things to do: exclude the NAns 
## or chnage them to zero
## try both and see what works better - presumably more data is good?

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
rf_model_R_rep0 = RandomForestRegressor(random_state=1)
rf_model_R_rep0.fit(train_X_rep0, train_y)
rf_val_pred_R_rep0 = rf_model_R_rep0.predict(val_X_rep0)
rf_val_mae_R_rep0 = mean_absolute_error(rf_val_pred_R_rep0, val_y)

## error with NAn values
## so do we just remove them or haow can be get rnadom forest to deal with them?
#%%

## ok so now we find what the erros in the orediction actually are
print("Validation MAE for Random Forest Model {:,.0f}".format(rf_val_mae_R_rep0))

## ok so it is marginally better
# $20,257

#%%
#############################################################################################
############################################################################################
# remove all Nans
# create new dataframe name to avoid confusion
## while dropping all nans

## doing this will also require a new y as there will be fewer rows of data - issue downstream

## first select data frame with the features plus y
features_R_filtNA = ["LotFrontage", "OverallQual", "YearRemodAdd", "MasVnrArea", "GrLivArea", "FullBath", "TotRmsAbvGrd", 
            "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF",  "SalePrice"]
X_NA = home_data[features_R_filtNA]

## drop the NAs
X_filt = X_NA.dropna(axis="rows")

X_filt.shape[0]
## 1195 rows 
## now we build the model
#%%
##define new y
y_filt = X_filt.SalePrice

#%%
# and correspondingly drop the variable from the X variable set
X_filt_drop = X_filt.drop(['SalePrice'], axis=1)


#%%
## now one way to do this is to split the home_data into a traing and test set
## not really needed here but done for properity at this time

train_X_filt, val_X_filt, train_y_filt, val_y_filt = train_test_split(X_filt_drop, y_filt, random_state=1)

#%%

## now build a random forest model using these four sets 
rf_model_R_filt = RandomForestRegressor(random_state=1)
rf_model_R_filt.fit(train_X_filt, train_y_filt)
rf_val_pred_R_filt = rf_model_R_rep0.predict(val_X_filt)
rf_val_mae_R_filt = mean_absolute_error(rf_val_pred_R_filt, val_y_filt)

## error with NAn values
## so do we just remove them or haow can be get rnadom forest to deal with them?
#%%

## ok so now we find what the erros in the orediction actually are
print("Validation MAE for Random Forest Model {:,.0f}".format(rf_val_mae_R_filt))

## ok so it is much better
# $11,713
