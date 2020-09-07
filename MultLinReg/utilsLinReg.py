import sklearn.metrics as metrics
import pandas as pd
import numpy as np

def repair_chrdata(df, tCol):
    ### Parameters:
    # df: input dataframe
    # tCol: targeted column label with NaN
    ### Output
    # df: repaired dataframe
    # word: string of related dataframe column with some records have NaN in targeted column
    # count: number of records fixed in the targeted column with NaN

    # work out number of NaN records need to fix
    dFrm = df[df[tCol].isnull()]
    count = len(dFrm)
    # work out the fill up string (most appearance) at targeted column for NULL
    tword = df[tCol].unique().tolist()
    # print(tword)
    wordLT = df[tCol].value_counts(dropna=False)
    word = ''
    wordCnt = 0
    for index, value in wordLT.items():
        print(f'[COUNT] Index: {index}, Value: {value}')
        if wordCnt < value:
            word = index
            wordCnt = value
    # print(word)
    # print(wordLT)
    # update the targeted NaN with the most frequent string
    mask = df[tCol].isnull()
    df.loc[mask, tCol] = word
    print(f'[REPAIR] "{tCol}" with string: {word}, Count: {count}')
    return df, word, count

# Repair a single number data column contained NaN with median value
def repair_numdata(df, tCol):
    ### Parameters:
    # df: input dataframe
    # tCol: targeted column label with NaN
    ### Output
    # df: repaired dataframe
    # medianVal: median value of related dataframe column with some records have NaN in targeted column
    # count: number of records fixed in the targeted column with NaN

    # work out number of NaN records need to fix
    dFrm = df[df[tCol].isnull()]
    count = len(dFrm)
    # work out the median value of the records from targeted column
    medianVal = df[tCol].median()
    # update the targeted NaN with the median value
    mask = df[tCol].isnull()
    df.loc[mask, tCol] = medianVal
    print(f'[REPAIR] "{tCol}" Median: {medianVal}, Count: {count}')
    return df, medianVal, count

### Work out the educated guess targets to repair dataframe with NaN in 'repair_rdata' function
def repair_target(df, tCol, rCol):
    ### Parameters:
    # df: input dataframe
    # tCol: targeted column label with NaN
    # rCol: related column label without NaN for educated guess
    ### Output
    # target: column value of related column that have NaN in targeted column
    repair = df[df[tCol].isnull()]
    # print(repair[[rCol, tCol]])
    target = sorted(repair[rCol].unique().tolist())
    print(f'[TARGET] {tCol} NaN target: {target}')
    return target

### Educated guess to repair dataframe column contained NaN with mean value of related
### dataframe column
def repair_rcdata(df, tCol, rCol, target):
    ### Parameters:
    # df: input dataframe
    # tCol: targeted column label with NaN
    # rCol: related column label without NaN for educated guess
    # target: column value of related column that have NaN in targeted column
    ### Output
    # df: repaired dataframe
    # meanVal: mean value of related dataframe column with some records have NaN in targeted column
    # count: number of records fixed in the targeted column with NaN

    ### Main coding
    # work out number of NaN records need to fix
    dFrm = df[df[tCol].isnull()]
    dFrm = dFrm[dFrm[rCol] == target]
    count = len(dFrm)
    # work out the mean value of the records from related column
    repair = df.loc[df[rCol] == target]
    meanVal = round(repair[tCol].mean(), 3)
    if np.isnan(meanVal):
        meanVal = np.float64(0)
    # update the targeted NaN with the calculated mean value of related records
    df[tCol] = df.apply(
        lambda row: meanVal if np.isnan(row[tCol]) & (row[rCol] == target)
                            else row[tCol], axis=1
    )
    print(f'[REPAIR] {tCol}({target}) Mean: {meanVal}, Count: {count}')
    return df, meanVal, count

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    # median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r-squared (r2): ', round(r2,4))
    print('mean_absolute_error (MAE): ', round(mean_absolute_error,4))
    print('mean_squared_error (MSE): ', round(mse,4))
    print('root_mean_squared_error (RMSE): ', round(np.sqrt(mse),4))
