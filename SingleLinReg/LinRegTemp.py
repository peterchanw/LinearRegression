import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import model_selection
from sklearn import linear_model
import sklearn.metrics as metrics

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r-squared (r2): ', round(r2,4))
    print('mean_absolute_error (MAE): ', round(mean_absolute_error,4))
    print('mean_squared_error (MSE): ', round(mse,4))
    print('root_mean_squared_error (RMSE): ', round(np.sqrt(mse),4))

### Demo Linear Regression
# y = mx + c
# F = 1.8*x + 32
# y = [1.8*F + 32 + random.randint(-3,3) for F in x]              # Fahrenheit

### Data Generation
x = list(range(0,20))                      # Celsius
# y = [1.8*F + 32 for F in x]              # Fahrenheit
y = [1.8*F + 32 + random.randint(-3,3) for F in x]              # Fahrenheit
print(f'X: {x}')
print(f'Y: {y}')
plt.plot(x,y, '-*r')
# plt.show()
# convert data format to input ML library
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

# split dataset for training dataset and test dataset and setup the model
xTrain, xTest, yTrain, y_Test = model_selection.train_test_split(x,y, test_size=0.2)
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)
print(f'Cofficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
# check accuracy of the model
accuracy = model.score(xTrain, yTrain)
print(f'Accuracy: {round(accuracy*100,2)}')
# Regression statistics #
y_pred = model.predict(xTest)
regression_results(y_Test, y_pred)

# prediction vs. actual plot
x = x.reshape(1,-1)[0]
m = model.coef_[0][0]
c = model.intercept_[0]
y = [m*F + c for F in x]
plt.plot(x,y, '-*g')
plt.show()
