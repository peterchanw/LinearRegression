import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from MultLinReg.utilsLinReg import repair_target, repair_rcdata, repair_numdata, repair_chrdata
from MultLinReg.utilsLinReg import regression_results
import statsmodels.api as sm

### LOAD DATA ###
country = 'CHN'
print('-'*20); print('LOAD DATA'); print('-'*20)
dataFrm = pd.read_csv('owid-covid-data.csv', sep=',')
dataF = dataFrm[dataFrm['iso_code'].isin([country])].copy()
print(dataF.head())
print(dataF.tail())
print('Original data columns: ', list(dataF.columns.values))
print('Original data records: ', len(dataF.index))

### CHECKING MISSING DATA ###
print('-'*20); print('CHECKING MISSING DATA'); print('-'*20)
print('#'*3); print(f'Before: \n{dataF.isnull().sum()}'); print('#'*3);

# CHECK 1: drop columns because of no data
# (i.e. number of nulls in a particular column = total dataframe records)
dropList = ['new_tests', 'total_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', \
            'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_per_case', \
            'positive_rate', 'tests_units', 'handwashing_facilities']
dataF = dataF.drop(columns=dropList)
print(f'CHECK 1 (Drop No Data Columns): {dropList}')

# CHECK 2: drop columns because of same value through the column or insignificant columns
dropList = ['hospital_beds_per_thousand', 'life_expectancy']
dataF = dataF.drop(columns=dropList)
print(f'CHECK 2 (Drop Insignificant Columns): {dropList}')

# CHECK 3: educated guess to repair data
# For example:
# repair = dataF[dataF['new_cases_smoothed'].isnull()]
# print(repair[['new_cases', 'new_cases_smoothed']])
# 'new_cases' = 0,15,17,27  for 'new_cases_smoothed' = NULL

print(f'CHECK 3 (Repair Data): .....')

# Method 1: repair dataframe column contained NaN with mean value of related dataframe column
target = repair_target(df=dataF, tCol='new_cases_smoothed', rCol='new_cases')
# target = [0.0, 15.0, 17.0, 27.0]
for i in range(len(target)):
    dataF, meanVal, count = repair_rcdata(df=dataF, tCol='new_cases_smoothed', rCol='new_cases', target=target[i])

target = repair_target(df=dataF, tCol='new_deaths_smoothed', rCol='new_deaths')
# target = [0.0]
for i in range(len(target)):
    dataF, meanVal, count = repair_rcdata(df=dataF, tCol='new_deaths_smoothed', rCol='new_deaths', target=target[i])

target = repair_target(df=dataF, tCol='new_cases_smoothed_per_million', rCol='new_cases_per_million')
# target = [0.0, 0.01, 0.012, 0.019]
for i in range(len(target)):
    dataF, meanVal, count = repair_rcdata(df=dataF, tCol='new_cases_smoothed_per_million', \
                                        rCol='new_cases_per_million', target=target[i])

target = repair_target(df=dataF, tCol='new_deaths_smoothed_per_million', rCol='new_deaths_per_million')
# target = [0.0]
for i in range(len(target)):
    dataF, meanVal, count = repair_rcdata(df=dataF, tCol='new_deaths_smoothed_per_million', \
                                        rCol='new_deaths_per_million', target=target[i])

# Method 2: repair NaN with median value in a particular column
dataF, medianVal, count = repair_numdata(dataF, 'stringency_index')

# Method 3: repair NaN with the most frequent word in a particular column
# dataF, word, count = repair_chrdata(dataF, 'continent')

print('#'*3); print(f'After: \n{dataF.isnull().sum()} '); print('#'*3)
print('Cleaned data columns: ', list(dataF.columns.values))
print('Cleaned data records: ', len(dataF.index))
# save the repaired dataframe to a CSV file
timeline = dataF['date'].copy()
dataF.to_csv('covid19_CN.csv', index=False)

### PREPROCESS DATA ###
print('-'*20); print('PREPROCESS DATA'); print('-'*20)
# Categorical boolean mask
categorical_feature_mask = dataF.dtypes == object
# print(categorical_feature_mask)
# filter categorical columns using mask and turn it into a list
categorical_cols = dataF.columns[categorical_feature_mask].tolist()
print(f'Categorical columns: {categorical_cols}')
# instantiate label encoder object
le = LabelEncoder()
# apply le on categorical feature columns
dataF[categorical_cols] = dataF[categorical_cols].apply(lambda col: le.fit_transform(col))
print(dataF[categorical_cols].head())
# print(dataF[categorical_cols].tail())

### ANALYSIS MULTIPLE REGRESSION MODEL FITTING ###
# Linear Regression: y = mX + c
# Multiple Regression: y = aX1 + bX2 + cX3 + ... + n
# Polynomial Regression: y = aX^1 + bX^2 + cX^3 + ....
print('-'*20); print('ANALYSIS MULTIPLE REGRESSION MODEL'); print('-'*20)
# "Stringency Index" a composite measure based on nine response indicators including
# school closures, workplace closures, and travel bans, rescaled to a value from 0 to 100 (100 = strictest)
X = np.array(dataF[['new_cases', 'new_deaths', 'stringency_index']]).reshape(-1,3)
# X = np.array(dataF[['new_cases', 'stringency_index']]).reshape(-1,2)
y = np.array(dataF['total_cases']).reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(X, y)
print(f'Accuracy: {round(model.score(X, y)*100, 3)} %')
print(f'Intercept: {model.intercept_} ')
print(f'Coefficients: {model.coef_} ')
print('\n')
# statsmodels for multiple regression model fitting
# X = sm.add_constant(dataF[['new_cases', 'new_deaths', 'stringency_index']])
# mregr = sm.OLS(y, X).fit()
# print(mregr.summary())

### TRAINING ###
testRatio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio)
print('-'*20); print('TRAINING MODEL'); print('-'*20)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print(f'Model Coefficient: {model.coef_}')
print(f'Model Intercept: {model.intercept_}')
accuracy = model.score(X_train, y_train)
# print('Accuracy: %.3f %%' % (accuracy*100))
print(f'Accuracy: {round((accuracy*100),3)} % \n')
# check the regression statistics
y_pred = model.predict(X_test)
regression_results(y_test, y_pred)

### PREDICTION ###
print('-'*20); print('PREDICTION PLOT'); print('-'*20)
### Prepare a timeline for plotting
timeln = pd.to_datetime(timeline)
day = timeln.dt.day.astype(str)
mth = timeln.dt.month_name().str.slice(stop=3)
yrs = timeln.dt.year.astype(str).str.slice(start=2)
xt = mth + yrs      # get the label of month+year array
# Find all the indexes of timeline for Day 1
xtk = []        # store the indexes for x-axis label location
xtkLabel = []   # store corresponding labels of month and year
xtkTarget = ['1', '2', '3', '4', '5']
xtkFlag = True
for i in range(0, len(timeln)):
    if day.array[i] in xtkTarget: # match Day 1/2/3/4/5 of the month for x-axis label location
        if xtkFlag:
            xtk.append(i)
            xtkLabel.append(xt.array[i])
            xtkFlag = False
    else:
        xtkFlag = True
# print(xtk)
# print(xtkLabel)

plt.figure(1, figsize=(8,5))
x = np.array(range(0, len(timeln))).reshape(-1,1)
y0 = np.array(dataF['total_cases']).reshape(-1,1)
y1 = model.predict(X)
plt.plot(x, y0)
plt.scatter(x, y0, s=15, c='red', marker='o', label='actual')
plt.scatter(x, y1, s=15, c='green', marker='x', label='predict')
plt.xticks(ticks=xtk, labels=xtkLabel, fontsize=7)
plt.yticks(np.arange(-10000, max(y0), 10000))
plt.xlabel('Date')
plt.ylabel('cases')
plt.grid()
plt.legend()
plt.title('Coronavirus Trend - China')
plt.show()