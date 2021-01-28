import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

housing = pd.read_csv('data/housing.csv')

# initial guess
b0, b1 = 0.1, 1
housing['Guess Response'] = b0 + b1 * housing['RM']
housing['Guess Error'] = housing['MEDV'] - housing['Guess Response']
plt.figure(figsize=(10, 10))
# plt.title('Sum of Sq. Error (SSE): {}'.format(((housing['Guess Error']) ** 2).sum()))
plt.scatter(housing['RM'], housing['MEDV'], color='g', label='Observed')
# plt.plot(housing['RM'], housing['Guess Response'], color='red', label='Guess Response')
plt.show()

# linear regression
formula = 'MEDV~RM'
model = smf.ols(formula=formula, data=housing).fit()
housing['Best Response'] = model.params[0] + model.params[1] * housing['RM']
housing['Best Error'] = housing['MEDV'] - housing['Best Response']
plt.figure(figsize=(6, 6))
plt.title('Sum of Sq. Error (SSE): {}'.format(((housing['Best Error']) ** 2).sum()))
plt.scatter(housing['RM'], housing['MEDV'], color='g', label='Observed')
plt.plot(housing['RM'], housing['Best Response'], color='yellow', label='Best Response')
plt.show()

print(model.summary())
