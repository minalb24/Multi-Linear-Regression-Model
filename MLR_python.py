# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:59:12 2019

@author: Mbirari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing Dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values

#categorial Variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:, 3] =labelencoder_X.fit_transform(X[:, 3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# Avoid the Dummy Variable Trap  consider all columns except first column with index 0
X=X[:,1:] 

# Spliting data into Train and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting Multiplle Linear Regression on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# regressor is object and LinearRegression is Class
# now fit this object to fit method
regressor.fit(X_train, y_train)

#Prediction of test set result

y_pred=regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# find optimal team of independent variable


# using once function we can create array of 1
# to avaoid datatype error convert matrix into integer

# this column associated with x0 =1, in multi linear regression equation.

# Step 1

import statsmodels.formula.api as sm 
X= np.append(arr=np.ones((50,1)).astype(int),values=X, axis=1)

# start backward elimination
# create metrix of optimal variable only contain independent variable have high impact on profit


# Step2 

X_opt=X[:,[0,1,2,3,4,5]] 
 

# matrix with all independent variables
# select significance level and p value is less that SL then independent variables will stay in model and
# if p ls above than SL then independent variables will remove from model


# OLS is simple ordinary Least square Model 
regressor_OLS=sm.OLS(endog=y,exog =X_opt).fit()

# Step 3

regressor_OLS.summary()


# Lower the P value more significant your independent variables
# P value of column 2 is 0.99 which is way above of significance level of 5%
# so remove x2

# Step 4 & Step 5

X_opt=X[:,[0,1,3,4,5]] 
regressor_OLS=sm.OLS(endog=y,exog =X_opt).fit()
regressor_OLS.summary()

# regressor_OLS.summary() run this statment in Python Console to get Summary of model 
# and P value details based on that take decision to remove independent variable column 

# remove x1 column as p value is higher than SL

X_opt=X[:,[0,3,4,5]] 
regressor_OLS=sm.OLS(endog=y,exog =X_opt).fit()
regressor_OLS.summary()

# remove x4 column

X_opt=X[:,[0,3,5]] 
regressor_OLS=sm.OLS(endog=y,exog =X_opt).fit()
regressor_OLS.summary()

# remove x5 column

X_opt=X[:,[0,3]] 
regressor_OLS=sm.OLS(endog=y,exog =X_opt).fit()
regressor_OLS.summary()

# Output :
# Hence R&D Spends column has strongest impact on Profit 