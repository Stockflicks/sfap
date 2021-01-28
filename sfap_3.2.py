import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# difference in statistics
sample1 = pd.DataFrame(np.random.normal(50, 10, size=30))
print('The Sample Mean is ', sample1[0].mean())
print('The Sample Variance is ', sample1[0].var(ddof=1))

# empirical distribution: sample vars
all_vars = []
for t in range(1000):
    sample2 = pd.DataFrame(np.random.normal(10, 5, size=30))
    all_vars.append(sample2[0].var())
df = pd.DataFrame()
df['var'] = all_vars
df['var'].hist(bins=100, figsize=(10, 5))
plt.show()

# empirical distribution: sample means
all_means = []
for t in range(1000):
    sample2 = pd.DataFrame(np.random.normal(10, 5, size=30))
    all_means.append(sample2[0].mean())
df = pd.DataFrame()
df['mean'] = all_means
population = pd.DataFrame(np.random.normal(10, 5, size=10000))
population[0].hist(bins=100, color="red", figsize=(10, 5), density=1)
df['mean'].hist(bins=100, color="blue", figsize=(10, 5), density=1)
plt.show()

# Sampling: General Distribution (may not be normal)
sample_size, sample_means = 2, []  # try different sample sizes
population_general = pd.DataFrame([100, -100, 0, -100, 100])
for t in range(10000):
    sample3 = population_general[0].sample(sample_size, replace=True)
    sample_means.append(sample3.mean())
df_general = pd.DataFrame()
df_general['mean'] = sample_means
df_general.hist(bins=100, figsize=(10, 5), density=1)
plt.show()
