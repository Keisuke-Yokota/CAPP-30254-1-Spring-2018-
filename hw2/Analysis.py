# CS122 W'18: Markov models and hash tables
# Keisuke Yokota

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### 1 Read Data
def read(filename):
    df = pd.read_csv(filename, header=0)
    df = df.rename(columns={'Unnamed: 0':'ID'})
    df = df.set_index('ID')
    return df

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
    '''