# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:25:23 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:12:29 2024

@author: user
"""


#%%

# first read in the necessary packages
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import category_encoders as ce

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
## all of these were p<0.001 when building a simple linear model of that variable describign sale price
## not a perfect way to do it but its a start

## producing continous variables is simple
c = ["LotFrontage", "OverallQual", "YearRemodAdd", "MasVnrArea", "GrLivArea", "FullBath", "TotRmsAbvGrd", 
            "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "SalePrice"]

#%%


## however there are some probelms later when including what seem to be important catagorical variables
# read thee in
cat_vars = pd.read_csv("C:/Users/user/Documents\\GitHub\\ML-development\\Iowa_house_competition_Kaggle\\important_characters.csv")
cat_vars

#%%
## now create a features list for these 
features_cat = list(cat_vars.imp_values_chr)

#%%

## so we have 19 variables which from the analyis in R (will try to repeat in Python at some point) that I have reason to belive are important 
## how do we deal with these?

#%%

## create a version of training dat with these variables
X_chr = home_data[features_cat] 
X_chr.describe()
#%%
X_chr.shape
# 1460 rows of 9 variables

#%% any NANs?
X_chr.isnull().sum()
## yeah quite a few 
## 'alley' has 1369/1460 with NAN

#%%
## for now I'll remove all those columns that have NANs and fouc only on the four that are complete
X_chr = X_chr.dropna(axis="columns")
X_chr.describe()
## so now we have four columns
# with between four and five levels of each of these character variables
#%%

## need to perform some sort of encoding to make this work
## one hot encoding seems popular?
# since this leads to an extra column for each level of the variable then lets just do one at a time?

one_hot = ce.OneHotEncoder(cols=col_names[:-1])

#%%

## seperate into training sets?

## make copices of the data
X_train_oh = X_train.copy()

X_test_oh = X_test.copy()

X_train_oh = one_hot.fit_transform(X_train_oh)

X_test_oh = one_hot.transform(X_test_oh)

X_train_oh.head()





#%%
## now select the columsn from the training data that fit those features
# call this X
X2 = home_data[]

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
