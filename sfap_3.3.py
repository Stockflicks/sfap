from scipy.stats import norm
import pandas as pd
import numpy as np
fb = pd.read_csv('data/facebook.csv')
fb['logReturn'] = np.log(fb['Close'].shift(-1)) - np.log(fb['Close'])
fb['logReturn'].hist(bins=50)
sample_size = fb['logReturn'].shape[0]
sample_mean = fb['logReturn'].mean()
sample_std = fb['logReturn'].std(ddof=1) / sample_size**0.5
left, right = 0.05, 0.95  # 90% confidence interval
interval_left = sample_mean + norm.ppf(left) * sample_std
interval_right = sample_mean + norm.ppf(right) * sample_std
print('90% confidence interval is:\n', (interval_left, interval_right))


