#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 10:38:43 2019

@author: chanti
"""

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
                          

#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''
#Splitting dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
'''

#Fitting Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


#Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


#Visualizing the L R results
plt.scatter(X, y, color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff -- LR')
plt.xlabel('PositionLevel')
plt.ylabel('Salary')
plt.show()

#Visualizing the P R results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or bluff -- PR')
plt.xlabel('PositionLevel')
plt.ylabel('Salary')
plt.show()

#Predicting the results using LR
X_test = np.arange(min(X), max(X), 0.5)
X_test = X_test.reshape(len(X_test), 1)
lin_reg.predict(X_test)

#Predicting the results using PR
lin_reg2.predict(poly_reg.fit_transform(X_test))





















                                                                                                                                                                                                                                                                                                                           
