import pandas as pd
fb = pd.read_csv("data/facebook.csv")
print(type(fb))
print(fb.head())
print(fb.shape)
print(fb.describe())
fb = fb.set_index(["Date"])
print(fb.loc['2015-01-01':'2015-12-31'])
