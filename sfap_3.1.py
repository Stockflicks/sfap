import pandas as pd
data = pd.DataFrame()
data['Population'] = [55, 49, 75, 25, 19, 29, 74, 26, 55, 30]
sample_replacement = data['Population'].sample(5, replace=True)
print("With Replacement:", sample_replacement.tolist())
sample_no_replacement = data['Population'].sample(5, replace=False)
print("No Replacement:", sample_no_replacement.tolist())

# sampling, with replacement vs. no replacement
sample_replacement = data['Population'].sample(5, replace=True)
print("With Replacement:", sample_replacement.tolist())
sample_no_replacement = data['Population'].sample(5, replace=False)
print("No Replacement:", sample_no_replacement.tolist())

# get mean and variance, for population and sample
print('Population mean and variance:',
      data['Population'].mean(), "and", round(data['Population'].var()))
a_sample = data['Population'].sample(10, replace=True)
print('Sample mean and variance:', a_sample.mean(), "and", round(a_sample.var()))

# degree of freedom
length = 600
sample_variance_collection_0 = \
    [data['Population'].sample(10, replace=True).var(ddof=0)
     for i in range(length)]
sample_variance_collection_1 = \
    [data['Population'].sample(10, replace=True).var(ddof=1)
     for i in range(length)]
print("Population Variance:", data['Population'].var(ddof=0))
print("Sample Variance (n):", pd.DataFrame(sample_variance_collection_0)[0].mean())
print("Sample Variance (n-1):", pd.DataFrame(sample_variance_collection_1)[0].mean())
