import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, linear_model, model_selection
import sklearn.metrics as metrics

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r-squared (r2): ', round(r2,4))
    print('mean_absolute_error (MAE): ', round(mean_absolute_error,4))
    print('mean_squared_error (MSE): ', round(mse,4))
    print('root_mean_squared_error (RMSE): ', round(np.sqrt(mse),4))

### LOAD DATA ###
print('-'*30); print('IMPORT DATA ...'); print('-'*30)
data = pd.read_csv('houses_to_rent.csv', sep = ',')
# print(data.head(5))
data = data [['city','rooms','bathroom','parking spaces','fire insurance','furniture','rent amount']]
# print(data.head(5))

### PROCESS DATA ###
# STEP 1: convert all non-number logic (e.g. "furniture/not furnished" to "1/0")
le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform(data['furniture'])
# STEP 2: convert all non-number values (e.g. R$8,000 to 8000)
# data['rent amount'] = data['rent amount'].map(lambda i: i[2:]))
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:].replace(',','')))
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',','')))
print(data.head(9))

### CHECK FOR ANY MISSING NUMBER OR Non-Numbers (NaN) ###
print('-'*30); print('CHECKING NULL DATA (NaN)'); print('-'*30)
print(data.isnull().sum())                  # check for NaN
# data = data.dropna()                      # drop records with NaN
# print(data.isnull().sum())                # check for NaN
print('-'*30); print('HEAD'); print('-'*30)
print(data.head())

### SPLIT DATA ###
print('-'*30); print('SPLIT DATA'); print('-'*30)
x = np.array(data.drop(['rent amount'],1))  # need all the inputs except the destinated output; axis=1 (column)
y = np.array(data['rent amount'])           # Machine Learning required numpy array format for input/output
print('X', x.shape)
print('Y', y.shape)
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size= 0.2, random_state=10)
print('xTrain', xTrain.shape)
print('xTest', xTest.shape)

### TRAIN MODEL ###
print('-'*30); print('TRAIN MODEL'); print('-'*30)
model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
accuracy = model.score(xTest, yTest)
print('Cofficients: ', model.coef_)
print('Intercept: ', model.intercept_)
print('Accuracy: %5.3f %%' % (accuracy*100))

### EVALUATION MODEL ###
# Regression statistics #
print('-'*30); print('MANUAL TEST & EVALUATION'); print('-'*30)
y_pred = model.predict(xTest)
regression_results(yTest, y_pred)
# Error statistics plots #
error=[]
for i, pred in enumerate(y_pred):
    error.append(yTest[i] - pred)

### box plotting ###

# plt.figure(1)
# plt.boxplot(error)
# plt.ylabel('pred error: R$')
# plt.title('rent amount')
# plt.show()

# plt.figure(2)
xCity = np.transpose(xTest)[0]      # City
xCity = ['City' if x==0 else 'Country' for x in xCity]
xFurn = np.transpose(xTest)[5]      # Furniture
xFurn = ['Furniture' if x==0 else 'Non-Furniture' for x in xFurn]
# sns.set(style="whitegrid")
# ax = sns.boxplot(x=xCity, y=error).set_title('rent amount')
# ax = sns.swarmplot(x=xCity, y=error, hue=xFurn, color='red')
# plt.xlabel('Metropolitan')
# plt.ylabel('pred error: R$')

### subplots ###
fig,axes = plt.subplots(1,2, figsize=(12,4))
sns.set(style="whitegrid")
axes[0].set(xlabel='data', ylabel='pred error: R$')
axes[1].set(xlabel='Metropolitan', ylabel='pred error: R$')
sns.boxplot(error, orient='v', ax=axes[0]).set_title('rent amount')
sns.boxplot(x=xCity, y=error, ax=axes[1]).set_title('rent amount')
sns.swarmplot(x=xCity, y=error, hue=xFurn, color='red', ax=axes[1])

### boxplot ###
plt.figure(2)
dfLabels = ['city','rooms','bathroom','parking spaces','fire insurance','furniture']
df = pd.DataFrame(xTest, columns=dfLabels)
df.loc[(df.city == 1),'city'] = 'City'
df.loc[(df.city == 0),'city'] = 'Country'
df.loc[(df.furniture == 1),'furniture'] = 'Furnitured'
df.loc[(df.furniture == 0),'furniture'] = 'Non-Furnitured'
df['pred_rent'] = y_pred.astype(int)
df['actual_rent'] = yTest.astype(int)
df['error'] = yTest.astype(int) - y_pred.astype(int)
print(df.head())
ax = sns.boxplot(x='rooms',y='error', hue='bathroom', data=df, palette='bright')
plt.xlabel('rooms')
plt.ylabel('pred error: R$')

### distribution plot ###
plt.figure(3)
x = df['fire insurance'].values
mean = df['fire insurance'].mean()
sns.distplot(x, color='blue')
plt.xlabel('fire insurance')
plt.ylabel('R$')
plt.axvline(mean, 0,1, color = 'red')

### linear regression plot ###
sns.lmplot(x='actual_rent',y='error', hue='bathroom', ci=False, data=df, col='rooms',
           height = 3, line_kws={'color':'red'}, col_wrap=4)
plt.xlabel('actual rent: R$')
plt.ylabel('pred error: R$')

### pair plot ###
g = sns.pairplot(df[['error','rooms','bathroom','parking spaces']], hue='parking spaces', height=2)

### heatmap plot ###
plt.figure(6)
pc = df[['error', 'rooms', 'bathroom', 'parking spaces']].corr()
sns.heatmap(pc, annot=True)
plt.show()
