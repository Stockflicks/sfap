import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

fb = pd.read_csv('data/facebook.csv')
# fb['Close'].plot()
# plt.show()
fb['logReturn'] = np.log(fb['Close'].shift(-1)) - np.log(fb['Close'])
# fb['logReturn'].plot()
# plt.show()
# fb['logReturn'].hist(bins=100)
# plt.show()
sample_mean = fb['logReturn'].mean()
sample_std = fb['logReturn'].std(ddof=1)
n = fb['logReturn'].shape[0]
z_hat = ((sample_mean-0) / (sample_std / n**0.5))
alpha = 0.05
z_left = norm.ppf(alpha / 2, 0, 1)
z_right = -z_left
print('Significance level:', alpha)
print('z left and right:', z_left, z_right)
print('z hat:', z_hat)
print('Reject H0' if z_hat > z_right or z_hat < z_left else 'Not Reject H0')
p = 1 - norm.cdf(abs(z_hat), 0, 1)
print('p value:', p)
print('Reject by p value' if p < alpha else 'No reject by p value')
