#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

die = pd.DataFrame([1, 2, 3, 4, 5, 6])
sum_of_dice = die.sample(2, replace=True).sum().loc[0]
print('Sum of dice is', sum_of_dice)  

np.random.seed(1)

trial = 50
result = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]

print(result[:10])

