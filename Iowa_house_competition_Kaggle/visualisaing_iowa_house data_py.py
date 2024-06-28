# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:08:41 2024

@author: user
"""

import pandas as pd



####################################################################################################

#### Iowa visualisation house data###

## the aim of this file is to repat what i did in R to find variables in the data that i think might be interesting 
## will help me learn the nuances between R and Python

#%%
# import the data
train_data = pd.read_csv('C:/Users/user/Documents/GitHub/ML-development/Iowa_house_competition_Kaggle/train.csv')

#%%
# have a look
train_data.head()

#%% 
# another look
train_data.describe()

#%%

# is there a str() equovaent in pyhtn?
train_data.info()