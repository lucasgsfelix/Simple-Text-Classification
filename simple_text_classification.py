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


models = {
            
            'LightGBM': LGBMClassifier(),
            'SVM': SVC(C=8.000e+00, kernel='linear'),
            'RF': RandomForestClassifier(class_weight='balanced'),
            'GB': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(),
            'LogisticRegression': LogisticRegression(class_weight='balanced')
        
         }


# In[45]:


complete_results = []

for model_name, model in tqdm.tqdm(models.items()):

    print("Model - ", model_name)
    
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


df_complete.to_csv("trip_advisor_general_models_results.csv", sep=';', index=False)
