# -*- coding: utf-8 -*-

###  script for Iowa house data 
## this will follow basic Kaggle submission
## further scripts will build on this model with insights from the R script

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

#%%

## now one way to do this is to split the home_data into a traing and test set
## not really needed here but done for properity at this time
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#%%

## now build a random forest model using these four sets 
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_pred = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_pred, val_y)

#%%

## ok so now we find what the erros in the orediction actually are
print("Validation MAE for Random Forest Model {:,.0f}".format(rf_val_mae))
## ok so th avergae preiction is $21,857 off

#%%

## we can improve this by makign he traing set bigger
## include all the training data and test against the test data









