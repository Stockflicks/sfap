#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

housing = pd.read_csv('housing.csv', index_col=0)
housing.head()

# Use this line to calculate covariance
housing.cov()

# Use this line to calculate correlation
housing.corr()

from pandas.plotting import scatter_matrix
sm = scatter_matrix(housing, figsize=(10, 10))

# MEDV vs LSTAT
housing.plot(kind='scatter', x='LSTAT', y='MEDV', figsize=(10, 10))


# In[ ]:




