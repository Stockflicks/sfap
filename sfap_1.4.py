import pandas as pd
import matplotlib.pyplot as plt
fb = pd.read_csv('data/facebook.csv')
fb = fb.set_index(["Date"])

fb['PriceDiff'] = fb['Close'].shift(-1) - fb['Close']
fb['Return'] = fb['PriceDiff'] / fb['Close']
fb['Direction'] = \
    [1 if fb['PriceDiff'].loc[ei] > 0 else 0 for ei in fb.index]
fb['ma10'] = fb['Close'].rolling(10).mean()
fb['ma50'] = fb['Close'].rolling(50).mean()

plt.figure(figsize=(10, 8))
fb['ma10'].loc['2015-01-01':'2015-12-31'].plot(label='MA10')
fb['ma50'].loc['2015-01-01':'2015-12-31'].plot(label='MA50')
fb['Close'].loc['2015-01-01':'2015-12-31'].plot(label='Close')
plt.title('Price difference on {} is {}. direction is {}'
          .format('2015-01-05', fb['PriceDiff'].loc['2015-01-05'],
                  fb['Direction'].loc['2015-01-05']))
plt.legend()
plt.show()



