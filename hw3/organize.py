import pandas as pd

filename1 = 'outcomes.csv'
filename2 = 'projects.csv'

o_df = pd.read_csv(filename1, header=0)
p_df = pd.read_csv(filename2, header=0)

df = p_df[(p_df.date_posted >= '2011-01-01')&(p_df.date_posted <= '2013-12-31')]
df.to_csv("project_2011_to_2013.csv")

u = df['state'].unique()
vc = o_df['fully_funded'].value_counts()

new_df = pd.merge(df, o_df, on='projectid', how='left')

new_df = df.drop(["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1.1"],axis=1) 



def logistic_regression_tuning(x_train, y_train, C, penalty, cv):
    '''
    Tune the parameters of the model to get Best one by setting
    target score as recall score.

    Inputs:
      x_train (dataframe)
      y_train (dataframe)
      cost: list of integer
      cv: integer   
    '''
    logistic_grid_param = {'C':cost,
                        'penalty':penalty}
    recall_scoring = make_scorer(recall_score,  pos_label=1)
    logistic_grid_search = GridSearchCV(
                            LogisticRegression(), 
                            logistic_grid_param, scoring=recall_scoring, cv=cv)
    logistic_grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(logistic_grid_search.best_params_))
    print('Best score: {:.3f}'.format(logistic_grid_search.best_score_))
    return logistic_grid_search.best_params_, logistic_grid_search.best_estimator_


def knn_tuning(x_train, y_train, weights, n_neighbors, algorithm, cv):
    '''
    Tune the parameters of the model to get Best one by setting
    target score as recall score.

    Inputs:
      x_train (dataframe)
      y_train (dataframe)
      weights: list of string such as 'uniform' or 'distance'
      n_neighbors: list of integers
      cv: integer
    '''    
    knn_grid_param = { 'weights': weights,
                       'n_neighbors': n_neighbors
                       'algorithm':algorithm}
    recall_scoring = make_scorer(recall_score,  pos_label=1)
    knn_grid_search = GridSearchCV(
                            KNeighborsClassifier(), 
                            knn_grid_param, scoring=recall_scoring, cv=cv)
    knn_grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(knn_grid_search.best_params_))
    print('Best score: {:.3f}'.format(knn_grid_search.best_score_))
    return knn_grid_search.best_params_, knn_grid_search.best_estimator_


def decision_tree_tuning(x_train, y_train
    , criterion, max_features, max_depth, min_samples_split, cv):
    tree_grid_param = {'criterion': criterion,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split}
    recall_scoring = make_scorer(recall_score,  pos_label=1)
    tree_grid_search = GridSearchCV(
                            DecisionTreeClassifier(), 
                            tree_grid_param, scoring=recall_scoring, cv=cv)
    tree_grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(tree_grid_search.best_params_))
    print('Best score: {:.3f}'.format(tree_grid_search.best_score_))
    return tree_grid_search.best_params_, tree_grid_search.best_estimator_



def random_forest_tuning(x_train, y_train
    , n_estimators, max_features, max_depth, min_samples_split, cv):
    forest_grid_param = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split}
    recall_scoring = make_scorer(recall_score,  pos_label=1)
    forest_grid_search = GridSearchCV(
                            RandomForestClassifier(random_state=0), 
                            forest_grid_param, scoring=recall_scoring, cv=cv,
                            n_jobs = -1)
    forest_grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(forest_grid_search.best_params_))
    print('Best score: {:.3f}'.format(forest_grid_search.best_score_))
    return forest_grid_search.best_params_, forest_grid_search.best_estimator_


def svm_tuning(x_train, y_train, kernel, C,cv):
    svm_grid_param = {'kernel':kernel,
                      'C':C}
    recall_scoring = make_scorer(recall_score,  pos_label=1)
    svm_grid_search = GridSearchCV(
                            SVC(random_state=0), 
                            svm_grid_param, scoring=recall_scoring, cv=cv)
    svm_grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(svm_grid_search.best_params_))
    print('Best score: {:.3f}'.format(svm_grid_search.best_score_))
    return svm_grid_search.best_params_, svm_grid_search.best_estimator_


def bagging_tuning(x_train, y_train, n_estimators , max_features, cv):
    '''
    Tune the parameters of the model to get Best one by setting
    target score as recall score.

    Inputs:
      x_train (dataframe)
      y_train (dataframe)
      n_estimators: list of integer
      cv: integer
    '''
    bagging_grid_param = {'n_estimators':n_estimators
                        'max_features': max_features}
    recall_scoring = make_scorer(recall_score,  pos_label=1)
    bagging_grid_search = GridSearchCV(
                            BaggingClassifier(random_state=0), 
                            bagging_grid_param, scoring=recall_scoring, cv=cv,
                            n_jobs = -1)
    bagging_grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(bagging_grid_search.best_params_))
    print('Best score: {:.3f}'.format(bagging_grid_search.best_score_))
    return bagging_grid_search.best_params_, bagging_grid_search.best_estimator_


def boosting_tuning(x_train, y_train, n_estimators, algorithm ,cv):
    '''
    Tune the parameters of the model to get Best one by setting
    target score as recall score.

    Inputs:
      x_train (dataframe)
      y_train (dataframe)
      n_estimators: list of integer
      cv: integer
    '''
    boosting_grid_param = {'n_estimators':n_estimators
                            'algorithm': algorithm}
    recall_scoring = make_scorer(recall_score,  pos_label=1)
    boosting_grid_search = GridSearchCV(
                            AdaBoostClassifier(random_state=0), 
                            boosting_grid_param, scoring=recall_scoring, cv=cv,
                            n_jobs = -1)
    boosting_grid_search.fit(x_train, y_train)
    print('Best parameters: {}'.format(boosting_grid_search.best_params_))
    print('Best score: {:.3f}'.format(boosting_grid_search.best_score_))
    return boosting_grid_search.best_params_, boosting_grid_search.best_estimator_

