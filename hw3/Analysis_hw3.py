# CAPP30122 Spr'18: ML for public policy
# Homework 3
# Keisuke Yokota

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from numpy import eye
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.externals.six import StringIO 
import pydotplus
from IPython.display import Image
from graphviz import Digraph
from sklearn.pipeline import Pipeline
import pickle
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel


### 1 Read Data
def load_data(filename='dis_discreted_hw3.csv'):
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
    df = df.drop(["Unnamed: 0"],axis=1) 
    return df


def make_labels(x):
    if x == 'f':
        return 1
    elif x == 't':
        return 0


def make_labels2(x):
    if x == 'low poverty':
        return 0
    elif x == 'moderate poverty':
        return 1
    elif x == 'high poverty':
        return 2        
    elif x == 'highest poverty':
        return 3

def make_labels3(x):
    if x == 'Grades PreK-2':
        return 0
    elif x == 'Grades 3-5':
        return 1
    elif x == 'Grades 6-8':
        return 2        
    elif x == 'Grades 9-12':
        return 3


def make_labels4(x):
    if x == 'Mrs.':
        return 0
    elif x == 'Ms.':
        return 1
    elif x == 'Mr.':
        return 2        
    elif x == 'Dr.':
        return 3


def make_labels5(x):
    if x == 'suburban':
        return 0
    elif x == 'urban':
        return 1
    elif x == 'rural':
        return 2        


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
    return df.iloc[:, 37].value_counts()


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


def standardize(df, column_list):
    '''
    Standardize value in each columns of the dataframe

    Inputs:
      dataframe
      column_list: list of column name

    Returns: dataframe
    '''
    sc = StandardScaler()
    x_train_std = sc.fit_transform(df.loc[:, column_list])
    x_train_std = pd.DataFrame(x_train_std)
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


def reduce_dimension(list_n_components, x_train):
    '''
    Reduce dimensionality of the data

    Inputs:
      list_n_components: list of integer
      x_train: dataframe

    '''
    lst = []
    for i in list_n_components:
        lsa = TruncatedSVD(n_components=i,random_state = 0)
        reduced_features = lsa.fit_transform(x_train) 
        lst.append(reduced_features)
        print("""After Reducing to # of {0},
                 sum of explained variance ratio is {1}""".format(
                    i,round((sum(lsa.explained_variance_ratio_)),2)))
    return lst


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


def feture_selection():
    rfc = RandomForestClassifier (n_estimator=100, n_jobs=-1)
    fs = SelectFromModel(rfc)
    return fs


### 5 Build Classifier
def split_data(x, y, threshhold):
    '''
    Split the dataframe into one for training and test
    without thinking

    Inputs:
      x: dataframe
      y: dataframe      
      threshhold: float

    Returns: dataframes
    '''     
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                         y,
                                                        test_size=threshhold,
                                                        random_state=0)
    return x_train, x_test, y_train, y_test


def pipeline(classifier):
    estimators = zip(["feature_selection", "pca", "classifier"], 
                    [fs, pca, classifier])
    pl = Pipeline(estimators)
    return pl


def logistic_regression(x_train, y_train, penalty='l1',
                        C=1, cv=5, best_parameter=False):
    '''
    Inputs:
      x_train: dataframe
      y_train: dataframe
      penalty: string either 'l1' or 'l2'
      C: integer or float
      best_parameter: boolean
    '''
    if not best_parameter:
        log_model = LogisticRegression(penalty=penalty,
                                        C=C)
        log_model.fit(x_train,y_train)
    else:
        best_params = best_parameter 
        log_model = LogisticRegression(best_parameter['C']
                                        ,best_parameter['penalty'])   
    evaluate(log_model, x_train, y_train, cv)
    coeff_df = DataFrame([x_train.columns, log_model.coef_[0]])
    print(coeff_df)


def knn(x_train, y_train, weights='distance', n_neighbors=1,
        algorithm='auto' , cv=5, best_parameter=False):
    '''
    Inputs:
      x_train: dataframe
      y_train: dataframe
      n_estimators: integer
      weights: integer
      algorithm: string such as 'auto','ball_tree'or 'kd_tree'
      best_parameter: boolean
    '''
    if not best_parameter:
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors
                                        , weights=weights
                                        , algorithm=algorithm)
        knn_model.fit(x_train,y_train)
    else:
        best_params = best_parameter 
        knn_model = LogisticRegression(best_parameter['n_neighbors']
                    , best_parameter['weights'], best_parameter['algorithm'])   
    evaluate(knn_model, x_train, y_train, cv)


def svm(x_train, y_train, kernel='linear', C=1, cv=5
        ,best_parameter=False):
    '''
    Inputs:
      x_train: dataframe
      y_train: dataframe
      kernel: string
      C: integer or float
      best_parameter: boolean
    '''
    if not best_parameter:
        svm_model = SVC(kernel=kernel, C=C, random_state=0)
        svm_model.fit(x_train,y_train)
    else:
        best_params = best_parameter 
        svm_model = SVC(best_parameter['kernel'], best_parameter['C']
                    ,random_state=0)   
    evaluate(svm_model, x_train, y_train,cv)


def decision_tree(x_train, y_train, criterion, max_features, max_depth,
                 min_samples_split, cv=5, best_parameter=False):
    '''
    Inputs:
      x_train: dataframe
      y_train: dataframe
      n_estimators: integer
      max_features: integer
      max_depth: integer
      min_samples_split: integer
      best_parameter: boolean
    '''
    if not best_parameter:
        tree = DecisionTreeClassifier(criterion=criterion,
                                      max_depth=max_depth,
                                      max_features=max_features,
                                      min_samples_split=min_samples_split)
        tree.fit(x_train, y_train)
    else:
        best_params = best_parameter
        tree = DecisionTreeClassifier(max_depth=best_params['max_depth'], 
                                max_features=best_params['max_features'], 
                                min_samples_split=best_params['min_samples_split'],
                                criterion=best_params['criterion'])
        tree.fit(x_train, y_train)
    evaluate(tree, x_train, y_train, cv)


def random_forest(x_train, y_train, n_estimators, max_features, max_depth,
                 min_samples_split, cv=5, best_parameter=False):
    '''
    Inputs:
      x_train: dataframe
      y_train: dataframe
      n_estimators: integer
      max_features: integer
      max_depth: integer
      min_samples_split: integer
      best_parameter: boolean
    '''
    if not best_parameter:
        forest = RandomForestClassifier(random_state=0, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_split=min_samples_split,
                                n_estimators=n_estimators)
        forest.fit(x_train, y_train)
    else:
        best_params = best_parameter
        forest = RandomForestClassifier(random_state=0, 
                                max_depth=best_params['max_depth'], 
                                max_features=best_params['max_features'], 
                                min_samples_split=best_params['min_samples_split'],
                                n_estimators=best_params['n_estimators'])
        forest.fit(x_train, y_train)
    evaluate(forest, x_train, y_train, cv)


def bagging(x_train, y_train,  n_estimators=300, max_features=1, cv=5,
            best_parameter=False):
    '''
    Inputs:
      x_train: dataframe
      y_train: dataframe
      n_estimators: integer
      max_features: integer
      best_parameter: boolean
    '''
    if not best_parameter:
        bagging_model = BaggingClassifier(n_estimators=n_estimators,
                                        max_features=max_features
                                        , n_jobs=-1 ,random_state=0)
        bagging_model.fit(x_train,y_train)
    else:
        best_params = best_parameter 
        bagging_model = BaggingClassifier(best_parameter['n_estimators'],
                                      best_parameter['max_features'],
                                      n_jobs=-1, random_state=0)   
    evaluate(bagging_model, x_train, y_train, cv)


def boosting(x_train, y_train, n_estimators=300, algorithm='SAMME',cv=5,
        best_parameter=False):
    '''
    Inputs:
      x_train: dataframe
      y_train: dataframe
      n_estimators: integer
      algorithm: string either SAMME' or 'SAMME.R'
      best_parameter: boolean
    '''
    if not best_parameter:
        boosting_model = AdaBoostClassifier(algorithm=algorithm,
                                           n_estimators=n_estimators,
                                            n_jobs=-1, random_state=0)
        boosting_model.fit(x_train,y_train)
    else:
        best_params = best_parameter 
        boosting_model = AdaBoostClassifier(best_parameter['n_estimators'],
                                            best_parameter['algorithm'],
                                            n_jobs=-1, random_state=0)   
    evaluate(boosting_model, x_train, y_train, cv)


def tuning(x_train, y_train, classifier, grid_param, cv=5,
         scoring=make_scorer(recall_score,  pos_label=1)):
    '''
    Tune the parameters of the model to get Best one by setting
    target score(default as recall score).

    Inputs:
      x_train (dataframe)
      y_train (dataframe)
      classifier: model (pipeline)
      grid_param: dictionary
      cv: integer
      scoring
    ''' 
    grid_search = GridSearchCV(
                            classifier, 
                            grid_param,
                            scoring=scoring,
                            cv=cv,
                            n_jobs = -1)
    grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(grid_search.best_params_))
    print('Best score: {:.3f}'.format(grid_search.best_score_))
    return grid_search.best_params_, grid_search.best_estimator_


def grid_parameters():
    '''
    set of grid_parameters for casual trial.
    
    e.g.
    'RF':RandomForestClassifier
    'LR':LogisticRegression
    'AB':AdaBoostClassifier
    'DT':DecisionTreeClassifier
    'SVM':SVM
    'KNN':KNeighborsClassifier
    'BAG':BaggingClassifier
    '''
    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': [None, 'sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear', 'rbf']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG' :{'n_neighbors': [1,5,10,25,50,100],'max_features': [1,2,5,10]}     
           }
    
    small_grid = { 
    'RF':{'n_estimators': [100, 10000], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': [None,'sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear', 'rbf']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG' :{'n_neighbors': [1,5,10,25,50,100],'max_features': [2,10]}     
           }
    return large_grid, small_grid


### 6 Evaluate Classifier
def evaluate(classifier, x_train, y_train, cv):
    '''
    Evaluate a classifier and identify important feature
    and show each score for evaluation

    Inputs:
      classifier (sklearn model)
      x_train (dataframe)
      y_train (dataframe)

    '''
    print('Baseline: {}'.format(y_train.mean))
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
    print('cross_validation: {:.3f}'.format(
                    cross_val_score(classifier, x_train, y_train, cv=cv)))
    plot_roc(classifier, x_train, y_train)
    plot_precision_recall(classifier, x_train, y_train)


def plot_feature_importance(classifier, x_train):
    values, names = zip(*sorted(
                    zip(classifier.feature_importances_,
                        x_train.columns)))
    length = len(x_train.columns)
    plt.figure(figsize=(length,length))
    plt.barh(range(len(names)), values, align='center')
    plt.yticks(range(len(names)), names)


def plot_decision_boundary(x_train, x_test, y_train, y_test, classifier,
                  label_name):
    x_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(x_combined
                        , y_combined
                        , clf=classifier
                        , test_idx=test_index)
    plt.title(label_name)
    plt.legend(loc='upper left')
    plt.show()


def draw_tree(best_estimator, pdf_name):
    dot_data = StringIO()
    tree.export_graphviz(best_estimator,out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(pdf_name)
    Image(graph.create_png())


def plot_roc(classifier, x_train, y_train):
    fpr, tpr, thresholds = roc_curve(y_train,
                                     classifier.predict(x_train))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.show()
    print("Area under the ROC curve : %f" % roc_auc)


def plot_precision_recall(classifier, x_train, y_train):
    precision, recall, thresholds = precision_recall_curve(
                                    y_train, classifier.predict(x_train))
    area = auc(recall, precision)
    plt.plot(precision, recall, label="Precision-Recall curve")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall example: AUC = %0.2f' % area)
    plt.legend(loc="lower left")
    plt.show()
    print("Area Under Curve: %0.2f" % area)


def temporal_validation(start_time, end_time, 
                        prediction_window, prediction_windows):
    '''
    I couldn't come up with any good ideas about temporal validation, 
    I would like to borrow Professor Rayid Ghani's function and modified it a little
    https://github.com/rayidghani/magicloops/blob/master/temporal_validate.py
    '''
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
#    prediction_windows = [6, 12]
#    update_window = 12
    lst = []

    for prediction_window in prediction_windows:
        test_end_time = end_time_date
        while (test_end_time >= start_time_date + 2 * relativedelta(months=+prediction_window)):
            test_start_time = test_end_time - relativedelta(months=+prediction_window)
            train_end_time = test_start_time  - relativedelta(days=+1) # minus 1 day
            train_start_time = train_end_time - relativedelta(months=+prediction_window)
            while (train_start_time >= start_time_date ):
                print(train_start_time,train_end_time,test_start_time,
                      test_end_time, prediction_window)
                train_start_time -= relativedelta(months=+prediction_window)
                # call function to get data
                train_set, test_set = extract_train_test_sets (train_start_time,
                                                               train_end_time,
                                                               test_start_time,
                                                               test_end_time)
                lst.append((train_set, test_set))
                # fit on train data
                # predict on test data
            test_end_time -= relativedelta(months=+update_window)
    return lst