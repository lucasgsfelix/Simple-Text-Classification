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

import tqdm

from sklearn.utils.class_weight import compute_class_weight


# In[4]:


df = pd.read_table("trip_advisor_dataset.csv", sep=';')


# In[5]:


df = df[['text', 'trip type']]


# In[6]:


df['review_clean'] = hero.clean(df['text'])


# In[7]:


vectorizer = TfidfVectorizer()


# In[8]:


X = vectorizer.fit_transform(df['review_clean'])


# In[9]:


replace_classes = {trip: index for index, trip in enumerate(df['trip type'].unique())}


# In[10]:


replace_classes = {trip: index for index, trip in enumerate(df['trip type'].unique())}

df['classes'] = df['trip type'].replace(replace_classes)


# In[44]:

balanced = False


if balanced == True:

    models = {
                
                'LightGBM': LGBMClassifier(class_weight='balanced'),
                'SVM': SVC(C=8.000e+00, kernel='linear', class_weight='balanced'),
                'RF': RandomForestClassifier(max_depth=5, n_jobs=5, class_weight='balanced'),
                'GB': GradientBoostingClassifier(n_estimators=10, max_depth=3),
                'XGBoost': XGBClassifier(n_jobs=5),
                'LogisticRegression': LogisticRegression(class_weight='balanced')
            
             }


    # In[45]:

    sample_weight = compute_class_weight(class_weight='balanced', classes=df['classes'].unique(), y=df['classes'])

    for label, class_name in enumerate(df['classes'].unique()):

        df.loc[df['classes'] == class_name, 'class_weight'] = sample_weight[label]

else:

    df['class_weight'] = 1

    models = {
                
                'LightGBM': LGBMClassifier(),
                #'SVM': SVC(C=8.000e+00, kernel='linear'),
                'RF': RandomForestClassifier(max_depth=5, n_jobs=5),
                'GB': GradientBoostingClassifier(n_estimators=10, max_depth=3),
                'XGBoost': XGBClassifier(n_jobs=5),
                'LogisticRegression': LogisticRegression()
            
             }


complete_results = []


for model_name, model in tqdm.tqdm(models.items()):

    print("Model - ", model_name)

    error = False

    if model in ['XGBoost', 'GB']:


        try:

            model_results = cross_validate(model,
                                       X, df['classes'],
                                       cv=5,
                                       scoring=('f1_micro', 'f1_macro'),
                                       n_jobs=5,
                                       fit_params={'sample_weight': df['class_weight'].values})


            model =+ ' Balanced'

        except:

            error = True



    if not (model in (['XGBoost', 'GB'])) or error == True:


        model_results = cross_validate(model,
                                       X, df['classes'],
                                       cv=5,
                                       scoring=('f1_micro', 'f1_macro'),
                                       n_jobs=5)



    df_results = pd.DataFrame(model_results)

    df_results['model'] = model_name
    
    complete_results.append(df_results)


# In[46]:


df_complete = pd.concat(complete_results)


# In[47]:


if balanced == True:

    df_complete.to_csv("TripAdvisor_balanced.csv", sep=';', index=False)

else:

    df_complete.to_csv("TripAdvisor_unbalanced.csv", sep=';', index=False)
