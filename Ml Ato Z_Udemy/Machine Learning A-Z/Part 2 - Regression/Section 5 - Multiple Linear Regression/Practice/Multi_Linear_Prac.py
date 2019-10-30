#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:36:02 2019

@author: chanti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
                          

#importing dataset
dataset = pd.read_csv("forestfires.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

'''month_replace_map = {'month': {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                                  'may': 5, 'jun': 6, 'jul': 7 , 'aug': 8 , 'sep': 9,'oct': 10,'nov': 11, 'dec': 12}}

day_replace_map = {'month': {'sun': 1, 'mon': 2, 'tue': 3, 'wed':4,'thu': 5,
                                  'fri': 6, 'sat': 7}}

dataset.replace(month_replace_map, inplace=True)
dataset.replace(day_replace_map, inplace=True)


dataset['month'] = dataset['month'].astype('category')
dataset['day'] = dataset['day'].astype('category')
print(dataset.dtypes)'''


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder()
X[:, 2] = LabelEncoder_X.fit_transform(X[:, 2])
#X[:, 3] = LabelEncoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X = X[:, 1:]


#Splitting dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

print('Coefficients: \n', regressor.coef_)

print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))

print('Variance score: %.2f' % regressor.score(X_test, y_test))

import statsmodels.api as sm
X = np.append(arr = np.ones((517,1)).astype(int), values = X, axis = 1)


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]]
X_Modeled = backwardElimination(X_opt, SL)


X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,27,28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [25]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()



###Its not working --- For this data set use another alogorithm like ExtraTreesRegressor












































