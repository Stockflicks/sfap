#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

fb = pd.read_csv('facebook.csv', index_col= 'Date')
fb['MA10'] = fb['Close'].rolling(10).mean() # fast
fb['MA50'] = fb['Close'].rolling(50).mean() # slow
fb = fb.dropna()

fb['Shares'] = [1 if fb.loc[ei, 'MA10']>fb.loc[ei, 'MA50'] else 0 for ei in fb.index]
fb['Close1'] = fb['Close'].shift(-1)
fb['Profit'] = [fb.loc[ei, 'Close1'] - fb.loc[ei, 'Close'] if fb.loc[ei, 'Shares']==1 else 0 for ei in fb.index]
fb['Profit'].plot()
plt.axhline(y=0, color='red')

fb['wealth'] = fb['Profit'].cumsum()

fb['wealth'].plot()
plt.title('Total money you win is {}'.format(fb.loc[fb.index[-2], 'wealth']))


# In[ ]:




