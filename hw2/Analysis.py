# CS122 W'18: Markov models and hash tables
# Keisuke Yokota

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### 1 Read Data
def load_data(fn1='cs-training.csv', fn2='cs-test.csv'):
    df1 = pd.read_csv(fn1)
    df1 = df1.rename(columns={'Unnamed: 0':'ID', 
                        'NumberOfTime30-59DaysPastDueNotWorse':'Less2months',
                       'NumberOfTimes90DaysLate':'More3months',
                       'NumberOfTime60-89DaysPastDueNotWorse':'Less3months'})
    df1 = df1.set_index('ID')
    df2 = pd.read_csv(fn2)
    df2 = df2.rename(columns={'Unnamed: 0':'ID', 
                        'NumberOfTime30-59DaysPastDueNotWorse':'Less2months',
                       'NumberOfTimes90DaysLate':'More3months',
                       'NumberOfTime60-89DaysPastDueNotWorse':'Less3months'})
    df2 = df2.set_index('ID')
    return df1, df2

### 2 Explore Data
def plot(dataframe, column_name):
    df = dataframe[column_name]
    sns.distplot(df,kde = True)
    plt.show()

    '''
    df.info()
    df.describe()
    df0 = df[df['SeriousDlqin2yrs']==0]
    df1 = df[df['SeriousDlqin2yrs']==1]

    df0 = df[df['SeriousDlqin2yrs']==0]
    df1 = df[df['SeriousDlqin2yrs']==1]
    #df0.info()
    #df1.info()
    ##Y = df.iloc[0:100, 0].values
    ##Y = np.where(y == 0, -1,1)
    X0 = df0.iloc[0:8000, [4,10]].values
    X1 = df1.iloc[0:8000, [4,10]].values
    plt.scatter(X0[:,0], X0[:,1], color='red', marker='o', label='ZERO')
    plt.scatter(X1[:,0], X1[:,1], color='blue', marker='x', label='ONE')
    plt.legend(loc='upper left')
    plt.show()


    sns.distplot(df.dropna(subset=['NumberOfDependents']).NumberOfDependents,kde = True)
plt.show()
    '''

### 3 Pre-process Data
from scipy import stats
def adj_outlier(dataframe,column_name):
    target = dataframe[column_name]
    q1 = stats.scoreatpercentile(target, 25)
    q3 = stats.scoreatpercentile(target, 75) 
    target_iqr = q3 - q1

    target_iqr_min = q1 - (target_iqr) * 1.5
    target_iqr_max = q3 + (target_iqr) * 1.5
    df = dataframe[target < target_iqr_max]
    return df

def outlier_iqr(df):
    for i in range(1,len(df.columns)):
        col = df.iloc[:,i]
        q1 = stats.scoreatpercentile(col, 25)
        q3 = stats.scoreatpercentile(col, 75) 
        iqr = q3 - q1
        outlier_min = q1 - (iqr) * 1.5
        outlier_max = q3 + (iqr) * 1.5
        col[col < outlier_min] = None
        col[col > outlier_max] = None
    return df


def outlier(dataframe):
    df = dataframe
    for column in df.columns:
        new_df = adj_outlier(df,column)
        df = new_df
    return df
    '''
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())
    '''

### 4
def discretize(SeriesDataframe, bins):
    cut_df, bins = pd.cut(SeriesDataframe, bins, retbins=True)  
    return  cut_df, bins


def binarize(SeriesDataframe):
    from numpy import eye
    target = np.array(df.iloc[:, 0])
    one_hot = np.eye(2)[target]
    return one_hot