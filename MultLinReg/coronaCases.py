import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

### LOAD DATA ###
data = pd.read_csv('coronaCases.csv', sep=',')
data = data[['id', 'cases']]
print('-'*20); print('LOAD DATA'); print('-'*20)
# print(data.head())

### PREPARE DATA ###
print('-'*20); print('PREPARE DATA'); print('-'*20)
x = np.array(data['id']).reshape(-1,1)
y = np.array(data['cases']).reshape(-1,1)
# plt.plot(y, '-m')
# # plt.show()

### GET FEATURES ###
# Linear Regression: y = mX + c
# Multiple Regression: y = aX1 + bX2 + cX3 + ... + n
# Polynomial Regression: y = aX^1 + bX^2 + cX^3 + ....
polyFeature = PolynomialFeatures(degree=3)
x = polyFeature.fit_transform(x)
# print(x)

### TRAINING ### (lack of data and thus avoid data splitting)
print('-'*20); print('TRAINING MODEL'); print('-'*20)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
# print('Accuracy: %.3f %%' % (accuracy*100))
print(f'Accuracy: {round((accuracy*100),3)} %')
y0 = model.predict(x)
# plt.plot(y, '-m', label='actual')
# plt.plot(y0, '--b', label='prediction')
# plt.title('Coronavirus Cases')
# plt.legend()
# plt.show()

### PREDICTION ###
print('-'*20); print('PREDICTION'); print('-'*20)
print(len(x))
days = 30
print(f'Prediction cases after {days} days: ', end=' ')
predAhead = len(x)+days
yPred = polyFeature.fit_transform([[predAhead]])
print(f'{round((model.predict(yPred)/1000000).item(),2)} Million')

x1 = np.array(range(1,predAhead)).reshape(-1,1)
y1 = model.predict(polyFeature.fit_transform(x1))
plt.plot(y, '-m', label='actual')
plt.plot(y0, '-b', label='predict')
plt.plot(y1, '--g', label='forecast')
plt.title('Coronavirus Cases')
plt.legend()
plt.show()