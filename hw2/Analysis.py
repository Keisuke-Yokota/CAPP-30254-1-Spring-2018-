# CAPP30122 Spr'18: ML for public policy
# Homework 2
# Keisuke Yokota

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from numpy import eye
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


### 1 Read Data
def load_data(filename='credit-data.csv'):
    '''
    Load data from a csv file and get dataframe.
    Besides, rename 'NumberOfTime30-59DaysPastDueNotWorse' as 'Less2months',
    'NumberOfTimes90DaysLate' as 'More3months', and 
    'NumberOfTime60-89DaysPastDueNotWorse' as 'Less3months'.

    Inputs:
      filename (string): a name of csv file

    Returns: dataframe
    '''
    df = pd.read_csv(filename, header=0)
    df = df.rename(columns={ 
                    'NumberOfTime30-59DaysPastDueNotWorse':'Less2months',
                    'NumberOfTimes90DaysLate':'More3months',
                    'NumberOfTime60-89DaysPastDueNotWorse':'Less3months'})
    df = df.set_index('PersonID')
    return df


### 2 Explore Data
def summary_x(df):
    '''
    Get summary statistics of dataframe.

    Inputs:
      dataframe

    Returns: summary stats table of the dataframe
    '''
    return df.iloc[:, 1:].describe()


def summary_y(df):
    '''
    Get summary statistics of variables in the dataframe

    Inputs:
      dataframe

    Returns: summary stats table of variables in the dataframe
    '''
    return df.iloc[:, 0].value_counts()


def corr(df):
    '''
    Get correlation of the pair of variables in the dataframe.

    Inputs:
      dataframe

    Returns: correlation table of the pair of variables in the dataframe.
    '''
    return df.iloc[:, 1:].corr()


def plot_corr(dataframe):
    '''
    Draw heat map for the correlation of each variable in the dataframe.

    Inputs:
      dataframe

    Returns: heat map
    '''
    length = len(dataframe.columns)
    plt.figure(figsize=(length,length))
    return sns.heatmap(dataframe.corr(), annot=True)


def count_null(df):
    '''
    Find NaN of each variable in the dataframe.

    Inputs:
      dataframe

    Returns: table
    '''
    return df.isnull().sum()


def show_dist(df, y_column_name):
    '''
    Get distribution of each variables and scattered map of 
    pairs of variables in the dataframe.

    Inputs:
      dataframe

    Returns: graph tables
    '''
    return sns.pairplot(df.dropna(how='any'), hue=y_column_name)


### 3 Pre-process Data
def fill_na(df):
    '''
    Use mean to fill in missing value in each columns of the dataframe

    Inputs:
      dataframe

    Returns: dataframe
    '''
    df = df.fillna(df.mean())
    return df


def standardize(df):
    '''
    Standardize value in each columns of the dataframe

    Inputs:
      dataframe

    Returns: dataframe
    '''
    sc = StandardScaler()
    x_train_std = sc.fit_transform(df.iloc[:, 1:])
    x_train_std = pd.DataFrame(x_train_std)
    x_train_std.index = x_train_std.index + 1
    x_train_std.columns = df.iloc[:, 1:].columns
    return x_train_std


def winsorize(df):
    '''
    Winzorize value in each columns of the dataframe

    Inputs:
      dataframe

    Returns: dataframe
    '''
    for i in range(1,len(df.columns)):
        col = df.iloc[:,i]
        q1 = stats.scoreatpercentile(col, 25)
        q3 = stats.scoreatpercentile(col, 75) 
        iqr = q3 - q1
        outlier_min = q1 - (iqr) * 1.5
        outlier_max = q3 + (iqr) * 1.5
        col[col < outlier_min] = None
        col[col > outlier_max] = None
        fill_na(col)
    return df


### 4 Generate feature/predictor
def discretize(SeriesDataframe, bins, labels=False):
    '''
    Discretize a continuous variable in the dataframe

    Inputs:
      dataframe
      bins (integer): the number for division
      labels (boolean)

    Returns: dataframe and numpy array
    '''
    cut_df, cut_bins = pd.cut(SeriesDataframe, bins, 
                            retbins=True, labels=labels)  
    return  cut_df, cut_bins


def dumminize(Dataframe):
    '''
    Take a categorical variable and create dummy variables from it.

    Inputs:
      dataframe

    Returns: dataframe
    '''
    dummy_df = pd.get_dummies(Dataframe, drop_first=True)
    return dummy_df


### 5 Build Classifier
def random_tree(x_train, y_train, best_parameter=False):
    '''
    Build a random tree as a classifier and evaluate

    Inputs:
      x_train (dataframe)
      y_train (dataframe)
      best_parameter (boolean)
    '''
    if not best_parameter:
        forest = RandomForestClassifier(random_state=1)
        forest.fit(x_train, y_train)
    else:
        best_params = best_parameter
        forest = RandomForestClassifier(random_state=1, 
                                max_depth=best_params['max_depth'], 
                                max_features=best_params['max_features'], 
                                min_samples_leaf=best_params['min_samples_leaf'],
                                n_estimators=best_params['n_estimators'])
        forest.fit(x_train, y_train)
    evaluate(forest, x_train, y_train)



def tuning(x_train, y_train):
    '''
    Tune the parameters of the model to get Best one by setting
    target score as recall score.

    Inputs:
      x_train (dataframe)
      y_train (dataframe)
    '''    
    forest_grid_param = {'n_estimators': [10],
                        'max_features': [1, 2, None],
                        'max_depth': [1, 5, 10, None],
                        'min_samples_leaf': [1, 2, 4,]}
    recall_scoring = make_scorer(recall_score,  pos_label=1)
    forest_grid_search = GridSearchCV(
                            RandomForestClassifier(random_state=1), 
                            forest_grid_param, scoring=recall_scoring, cv=4)
    forest_grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(forest_grid_search.best_params_))
    print('Best score: {:.3f}'.format(forest_grid_search.best_score_))
    return forest_grid_search.best_params_


### 6 Evaluate Classifier
def evaluate(classifier, x_train, y_train):
    '''
    Evaluate a classifier and identify important feature
    and show each score for evaluation

    Inputs:
      classifier (sklearn model)
      x_train (dataframe)
      y_train (dataframe)

    '''
    values, names = zip(*sorted(
                    zip(classifier.feature_importances_,
                        x_train.columns)))
    length = len(x_train.columns)
    plt.figure(figsize=(length,length))
    plt.barh(range(len(names)), values, align='center')
    plt.yticks(range(len(names)), names)
    print('Train score: {}'.format(
                    classifier.score(x_train, y_train)))
    print('accuracy score: {:.3f}'.format(
                    accuracy_score(y_train, classifier.predict(x_train))))
    print('precison score: {:.3f}'.format(
                    precision_score(y_train, classifier.predict(x_train))))
    print('recall score: {:.3f}'.format(
                    recall_score(y_train, classifier.predict(x_train))))
    print('Confusion matrix:\n{}'.format(
                    confusion_matrix(y_train, classifier.predict(x_train))))
    print('f1 score: {:.3f}'.format(
                    f1_score(y_train, classifier.predict(x_train))))