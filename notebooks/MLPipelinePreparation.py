#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[346]:


#import the nltk module
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


# In[347]:


# import libraries
import pandas as pd
import numpy as np
import re
import time
from sqlalchemy import create_engine
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
import pickle


# In[371]:


# load data from database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)


# #### Print the first few rows from the dataframe

# In[372]:


df.head()


# #### Lets check  if there are any columns with null values

# In[373]:


df.dtypes


# In[374]:


df.isnull().sum()


# We see that none of the target categories have missing values.

# In[376]:


target_cols = list(df.select_dtypes(include='int64').columns)


# In[377]:


target_cols.remove('id')


# #### Lets check how many classes exist for each of the target categories

# In[363]:


for col in target_cols:
    display(df[col].value_counts())


# We see that the 'child_alone' category has only one class (0).<BR>
# We will delete the 'child_alone' category

# In[378]:


df.drop(columns='child_alone', inplace=True)


# In[380]:


target_cols.remove('child_alone')


# In[381]:


target_cols


# In[382]:


#Define the feature and target variables
X = df['message'].values
Y = df[target_cols].values


# In[383]:


display(X.shape); display(Y.shape)


# ### 2. Write a tokenization function to process your text data

# In[384]:


def tokenize(text):
    # Normalize text
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[18]:


pipeline = Pipeline([
       ('vect', CountVectorizer(tokenizer=tokenize)),
       ('tfidf', TfidfTransformer()),
       ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=2))),
    ]) 


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.3)

display(X_train.shape); display(Y_train.shape)
display(X_train[0:5])


# In[20]:


display(X_train[15:20])
display(Y_train[15:20])


# In[237]:


display(time.time())
pipeline.fit(X_train, Y_train)
display(time.time())


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[238]:


display(time.time())
Y_pred = pipeline.predict(X_test)
display(time.time())


# In[250]:


for i in range(len(target_cols)):
    cr = classification_report(Y_test[:,i], Y_pred[:,i], zero_division=1)
    print(target_cols[i])
    print(cr)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[271]:


pipeline.get_params().keys()


# In[272]:


parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 200]
    }
cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[273]:


display(time.time())
cv.fit(X_train, Y_train)
display(time.time())


# In[274]:


cv.best_params_


# In[275]:


display(time.time())
Y_pred = cv.predict(X_test)
display(time.time())


# In[276]:


for i in range(len(target_cols)):
    cr = classification_report(Y_test[:,i], Y_pred[:,i], zero_division=1)
    print(target_cols[i])
    print(cr)


# In[279]:


best_model = cv.best_estimator_


# In[280]:


display(time.time())
Y_pred = best_model.predict(X_test)
display(time.time())


# In[281]:


for i in range(len(target_cols)):
    cr = classification_report(Y_test[:,i], Y_pred[:,i], zero_division=1)
    print(target_cols[i])
    print(cr)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# #### Create a StartingVerbExtracter class to identify if the first word of a sentence is a verb, and return True if it is a verb.
# #### The StartingVerbExtracter will be used as an additional step in the Pipeline object to extract this additional feature.

# In[385]:


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# ###### We will use the iterative_train_test_split method from the skmultilearn.model_selection to perform a stratified sampling of the multi-labeled data

# In[386]:


X_train1, Y_train1, X_test1, Y_test1 = iterative_train_test_split(
    df[["message"]].values,
    df[target_cols].values,
    test_size = 0.3
)


# In[387]:


X_train1 = X_train1.flatten()
X_test1 = X_test1.flatten()


# ##### We will use the IterativeStratification method from the skmultilearn.model_selection to perform a stratified sampling when creating the k-folds during cross-validation.

# In[389]:


k_fold = IterativeStratification(n_splits=3, order=1)


# In[390]:


pipeline2 = Pipeline([
    ('features', FeatureUnion([

        ('nlp_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())]
        )),

        ('start_verb', StartingVerbExtractor())
     ])),
        
        ('clf', MLPClassifier(max_iter=300, random_state=42, verbose=True))
    ])


# In[391]:


parameters = {
        'clf__hidden_layer_sizes': [(30,), (50,)]
    }
grid_cv = GridSearchCV(pipeline2, param_grid=parameters, scoring='f1_weighted', cv=k_fold, n_jobs=-1)


# In[392]:


display(time.time())
grid_cv.fit(X_train1, Y_train1)
display(time.time())


# In[393]:


model = grid_cv.best_estimator_
grid_cv.best_params_


# In[394]:


display(time.time())
Y_pred1 = model.predict(X_test1)
display(time.time())


# ##### Report the f1 score, precision and recall for each output category of the dataset. We will do this by iterating through the columns and calling sklearn's classification_report on each.

# In[395]:


for i in range(len(target_cols)):
    cr = classification_report(Y_test1[:,i], Y_pred1[:,i], zero_division=1)
    print(target_cols[i])
    print(cr)


# ### 9. Export your model as a pickle file

# In[397]:


pkl_filename = "../models/model_classifier.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




