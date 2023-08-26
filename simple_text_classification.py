#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import texthero as hero

import numpy as np

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_validate

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

import tqdm

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import KFold


def tested_models():

    return {
                'LightGBM': LGBMClassifier, # class_weight='balanced'
                'RF': RandomForestClassifier, # class_weight='balanced'
                'LogisticRegression': LogisticRegression, # class_weight='balanced'
                'GB': GradientBoostingClassifier,
                'XGBoost': XGBClassifier
           }


def retrieve_best_params(grid_results):

    params = grid_results[grid_results['mean_test_score'] == grid_results['mean_test_score'].max()]['params'].values[0]

    best_params = {}

    # os parâmetros dos modelos começam com 'model__'
    for key, value in params.items():

        best_params[str(key).replace('model__', '')] = value

    return best_params


def tested_parameters(model):

    return {
                "XGBoost": {
                                'model__max_depth': [2, 3, 5, 7],
                                'model__n_estimators': [10, 50, 100, 200],
                                'model__gamma': [0, 0.1],
                                'model__eta': [0.1, 0.2],
                                'model__min_child_weight': [1, 2, 5],
                                'model__eval_metric': ['logloss']
                           },
                "LightGBM": {
                                'model__boosting_type': ['gbdt', 'dart', 'rf'],
                                'model__objective': ['binary'],
                                'model__njobs': [-1]
                           },
                "RF": {
                        'model__n_estimators': [10, 50, 100, 200],
                        'model__criterion': ['gini', 'entropy', 'log_loss'],
                        'model__max_features': ['sqrt', 'log2', None],
                        'model__n_jobs': [-1]

                      },
                "GB": {
                        'model__loss': ['log_loss', 'exponential'],
                        'model__n_estimators': [10, 50, 100, 200],
                        'model__criterion': ['friedman_mse', 'squared_error'],
                        'model__max_features': ['sqrt', 'log2', None],
                        'model__tol': [1e-6, 1e-4, 1e-2, 1],
                      },
                "LogisticRegression": {
                                            'model__penalty': [None, 'l2'],
                                            'model__C': [0.00001, 0.001, 0.1, 0.5, 1, 2, 10],
                                            'model__tol': [1e-6, 1e-4, 1e-2, 1],
                                            'model_fit_intercept': [True, False],
                                            'model__solver': ['sag', 'saga'],
                                            'model__max_iter': [10, 50, 100, 150, 1000],
                                            'model__n_jobs': [-1]
                                      }

           }[model]


def measure_class_weights(df):

    sample_weight = compute_class_weight(class_weight='balanced', classes=df['classes'].unique(), y=df['classes'])

    for label, class_name in enumerate(df['classes'].unique()):

        df.loc[df['classes'] == class_name, 'class_weight'] = sample_weight[label]

    return df


if __name__ == '__main__':


    df = pd.read_table("trip_advisor_dataset.csv", sep=';')

    df_yelp = pd.read_table("manual_reviews.csv", sep=';')

    df['dataset'] = 'TripAdvisor'

    df_yelp['dataset'] = 'Yelp'

    size_yelp, size_tripadvisor = len(df_yelp), len(df)

    df = pd.concat([df_yelp, df])

    df = df[['text', 'trip type']]

    df['review_clean'] = hero.clean(df['text'])

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(df['review_clean'])

    replace_classes = {trip: index for index, trip in enumerate(df['trip type'].unique())}

    df['classes'] = df['trip type'].replace(replace_classes)


    df = measure_class_weights(df)

    complete_results = []

    for train_dataset in ['Yelp', 'TripAdvisor']:
        
        if train_dataset == 'Yelp':
        
            # treinando com yelp e testando com trip advsior
            x_train, y_train = X[: size_yelp], df['classes'][: size_yelp]
            x_test, y_test = X[-size_tripadvisor:], df['classes'][-size_tripadvisor: ]
        
        else:
        
            x_train, y_train = X[-size_tripadvisor:], df['classes'][-size_tripadvisor: ]
            x_test, y_test = X[: size_yelp], df['classes'][: size_yelp]

        complete_results, parameters = [], []

        for balanced in [True, False]:

            for model_name, model in tqdm.tqdm(tested_models().items()):

                model_parameters = tested_parameters(model_name)

                if balanced is True and not (model_name in ['GB', 'XGBoost']):

                    model_parameters['model__class_weight'] = ['balanced']

                pipe = Pipeline(steps=[("model", model)])

                grid = GridSearchCV(estimator=pipe,
                                    param_grid=[model_parameters],
                                    cv=KFold(n_folds=5),
                                    scoring=('f1_micro', 'f1_macro'),
                                    return_train_score=True,
                                    n_jobs=-1)

                if balanced is True and model_name in ['GB', 'XGBoost']:

                    grid.fit(x_train, y_train, sample_weight=df['class_weight'])

                grid_results = pd.DataFrame(grid.cv_results_)

                best_parameters = retrieve_best_params(grid_results)

                grid_results['Model'] = model

                grid_results['Balanced'] = balanced

                prediction = model(**best_parameters).predict(x_test)

                results = {
                        'Model': model,
                        'Balanced': balanced,
                        'f1-micro': f1_score(y_test, prediction, average='micro'),
                        'f1-macro': f1_score(y_test, prediction, average='macro'),
                        'f1-binary': f1_score(y_test, prediction, average='binary'), 
                        'accuracy': accuracy_score(y_test, prediction)
                      }
                
                complete_results.append(pd.DataFrame([results]))

                parameters.append(grid_results)

        df_complete = pd.concat(complete_results)

        df_parameters = pd.concat(parameters)

        df_complete.to_csv(dataset + ".csv", sep=';', index=False)

        df_parameters.to_csv(dataset + "_parameters.csv", sep=';', index=False)
