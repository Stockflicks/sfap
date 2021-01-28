import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings("ignore")

aor = pd.read_csv('data/indice/ALLOrdinary.csv').set_index('Date')
cac = pd.read_csv('data/indice/CAC40.csv').set_index('Date')
dax = pd.read_csv('data/indice/DAXI.csv').set_index('Date')
dji = pd.read_csv('data/indice/DJI.csv').set_index('Date')
hsi = pd.read_csv('data/indice/HSI.csv').set_index('Date')
nas = pd.read_csv('data/indice/nasdaq_composite.csv').set_index('Date')
nik = pd.read_csv('data/indice/Nikkei225.csv').set_index('Date')
sap = pd.read_csv('data/indice/SP500.csv').set_index('Date')
spy = pd.read_csv('data/indice/SPY.csv').set_index('Date')

# print(nas.head())

panel = pd.DataFrame(index=spy.index)
panel['aord'] = aor['Close'] - aor['Open']
panel['cac40'] = cac['Open'] - cac['Open'].shift(1)
panel['daxi'] = dax['Open'] - dax['Open'].shift(1)
panel['dji'] = dji['Open'] - dji['Open'].shift(1)
panel['hsi'] = hsi['Close'] - hsi['Open']
panel['nasdaq'] = nas['Open'] - nas['Open'].shift(1)
panel['nikkei'] = nik['Close'] - nik['Open']
panel['sp500'] = sap["Open"] - sap['Open'].shift(1)
panel['spy'] = spy['Open'].shift(-1) - spy['Open']
panel['spy_lag1'] = panel['spy'].shift(1)
panel['Price'] = spy['Open']
# print(panel.head())
# print(panel.isnull().sum())
panel = panel.fillna(method='ffill').dropna()
print(panel.head())
panel.to_csv('data/indice/indicepanel.csv', mode="w")

# print(panel.shape)
Train = panel.iloc[-2000:-1000, :]
Test = panel.iloc[-1000:, :]
# print(Train.shape, Test.shape)

# scatter_matrix(Train, figsize=(10, 10))
# plt.show()
corr_array = Train.iloc[:, :-1].corr()['spy']
# print(corr_array)
lm = smf.ols(formula='spy~spy_lag1+sp500+nasdaq+dji+cac40+aord+daxi+nikkei+hsi', data=Train).fit()
print(lm.summary())

Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)
# plt.scatter(Train['spy'], Train['PredictedY'])
# plt.show()

print(Train.iloc[:, :-1].corr())

def adjMetric(df, model, model_k, y_name):
    df['yhat'] = model.predict(df)  # y-hat
    SST = ((df[y_name] - df[y_name].mean()) ** 2).sum()
    SSR = ((df['yhat'] - df[y_name].mean()) ** 2).sum()
    SSE = ((df[y_name] - df['yhat']) ** 2).sum()
    return 1 - (1 - SSR / SST) * (df.shape[0] - 1) / (df.shape[0] - model_k - 1), \
           (SSE / (df.shape[0] - model_k - 1)) ** 0.5

def assessTable(test, train, model, model_k, y_name):
    r2test, RMSE_test = adjMetric(test, model, model_k, y_name)
    r2train, RMSE_train = adjMetric(train, model, model_k, y_name)
    asm = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    asm['Train'], asm['Test'] = [r2train, RMSE_train], [r2test, RMSE_test]
    return asm
print(assessTable(Test, Train, lm, 9, 'spy'))
