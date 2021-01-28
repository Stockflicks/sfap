#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ms = pd.read_csv('microsoft.csv', index_col= 'Date')

ms['LogReturn'] = np.log(ms['Close']).shift(-1) - np.log(ms['Close'])
from scipy.stats import norm
mu = ms['LogReturn'].mean()
sigma = ms['LogReturn'].std(ddof=1)

density = pd.DataFrame()
density['x'] = np.arange(ms['LogReturn'].min()-0.01, ms['LogReturn'].max()+0.01, 0.001)
density['pdf'] = norm.pdf(density['x'], mu, sigma)

ms['LogReturn'].hist(bins=50, figsize=(15, 8))
plt.plot(density['x'], density['pdf'], color='red')
plt.show()

prob_return1 = norm.cdf(-0.05, mu, sigma)
print('The Probability is ', prob_return1)

mu220 = 220*mu
sigma220 = (220**0.5) * sigma
print('The probability of dropping over 40% in 220 days is ', norm.cdf(-0.4, mu220, sigma220))

VaR = norm.ppf(0.05, mu, sigma)
print('Single day value at risk ', VaR)

print('5% quantile ', norm.ppf(0.05, mu, sigma))

# Quatile 
# 5% quantile
print('5% quantile ', norm.ppf(0.05, mu, sigma))
# 95% quantile
print('95% quantile ', norm.ppf(0.95, mu, sigma))


# In[ ]:




