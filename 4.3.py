#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Data preparation
housing = pd.read_csv('housing.csv')
housing = housing.set_index('Unnamed: 0')

model = smf.ols(formula='MEDV~LSTAT', data=housing).fit()

# Here are estimated intercept and slope by least square estimation 
b0_ols = model.params[0]
b1_ols = model.params[1]

housing['BestResponse'] = b0_ols + b1_ols*housing['LSTAT']

#Linearity

housing.plot(kind='scatter', x='LSTAT', y='MEDV', figsize=(10, 10), color='g')

#Independence

# Get all errors (residuals)
housing['error'] = housing['MEDV'] - housing['BestResponse']
# Method 1: Residual vs order plot
# error vs order plot (Residual vs order) as a fast check 
housing['error'] = housing['MEDV'] - housing['BestResponse']
plt.figure(figsize=(15, 8))
plt.title('Residual vs order')
plt.plot(housing.index, housing['error'], color='purple')
plt.axhline(y=0, color='red')
plt.show()
# Method 2: Durbin Watson Test
# Check the Durbin Watson Statistic
# Rule of thumb: value in the range of 1.5 to 2.5 are relatively normal
model.summary()

#Normality

import scipy.stats as stats
z = (housing['error'] - housing['error'].mean())/housing['error'].std(ddof=1)

stats.probplot(z, dist='norm', plot=plt)
plt.title('Normal Q-Q plot')
plt.show()

#Equal variance

# Residual vs predictor plot
housing.plot(kind='scatter', x='LSTAT', y='error', figsize=(15, 8), color='green')
plt.title('Residual vs predictor')
plt.axhline(y=0, color='red')
plt.show()


# Conclusion:We can see that the regression model (MEDV~LSTAT) violates all four assumptions. 
# Therefore, we cannot make statistical inference using this model.

