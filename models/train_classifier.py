#import the nltk module
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

#import the libraries
import sys
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
import pickle

def load_data(database_filepath):
    """
    The function connects to a sqlite database and loads the disaster_response
    table to a pandas dataframe. It extracts the messages, the target categories 
    tagged with each of the messages, and the category column names.

    Parameters:
    database_filepath (str) : path to the sqlite database

    Returns:
    X (numpy.ndarray) : numpy array  containing the messages 
    Y (numpy.ndarray) : numpy array containing the categories associated with the messages
    category_names (list) : list containing all the category names
    """

    db_connect_str = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(db_connect_str)
    df = pd.read_sql_table('disaster_response', engine)

    target_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)

    #drop target categories which have one or less class values 
    for col in target_cols:
        if len(list(df[col].value_counts().index)) <= 1: 
            df.drop(columns=col, inplace=True)

    target_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)

    #remove the 'id' column from the target_cols list
    try:
        target_cols.remove('id')
    except ValueError:
        pass

    X = df[['message']].values
    Y = df[target_cols].values
    category_names = target_cols
    print(X[:5])
    print(Y[:5])
    print(category_names)

    return X, Y, category_names


def tokenize(text):
    """
    The function normalizes the text by converting it to lower case, it uses 
    the nltk package to tokenize the text to word tokens, and lemmatize the tokens. 

    Parameters:
    text (str) : the text which needs to be parsed into tokens

    Returns:
    list : the list of parsed tokens
    """

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


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """ 
    The class StartingVerbExtractor is used to detect if the starting
    token in the text(message) is a verb or not.

    The function starting_verb of this class uses the nltk package to
    implement this check. If the starting token is a verb, it returns 
    True else returns False.
    """

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


def build_model():
    """
    The function uses the Pipeline and GridSearchCV objects along 
    with cross-validation to build the model.

    It uses the IterativeStratification method from the skmultilearn.model_selection 
    to perform a stratified sampling when creating the k-folds for cross-validation.

    The Pipeline is built using CountVectorizer, TfidfTransformer to vectorize the text,
    and it uses the StartingVerbExtractor to detect if the starting token is a verb. 
    
    The classifier used is the MLPClassifier.

    Parameters:
    none

    Returns:
    GridSearchCV : model built using GridSearchCV 
    """

    k_fold = IterativeStratification(n_splits=3, order=1)
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('nlp_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())]
        )),

        ('start_verb', StartingVerbExtractor())
     ])),
        
        ('clf', MLPClassifier(max_iter=300, random_state=42, verbose=True))
    ])

    parameters = {
        'clf__hidden_layer_sizes': [(30,), (50,)]
    }
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_weighted', cv=k_fold, n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function evaluates the model built using GridSearchCV, and prints the  
    classification report depicting f1 score, precision, recall for each of the 
    output categories.

    Parameters:
    model (GridSearchCV) : the model built using GridSearchCV which needs to be evaluated
    X_test (numpy.ndarray) : the array containing the messages in the test dataset 
    Y_test (numpy.ndarray) : the array containig the categories associated with each of the test messages 
    category_names (list) : the list of output category names

    Returns:
    none
    """

    best_model = model.best_estimator_
    Y_pred = best_model.predict(X_test)

    for i in range(len(category_names)):
        cr = classification_report(Y_test[:,i], Y_pred[:,i], zero_division=1)
        print(category_names[i])
        print(cr) 
    


def save_model(model, model_filepath):
    """
    The function uses the python module pickle to save the model to a file.

    Parameters:
    model (GridSearchCV) : the model which needs to be saved
    model_filepath (str) : the OS file path to where the model needs to be saved

    Returns:
    none
    """

    best_model = model.best_estimator_
    with open(model_filepath, 'wb') as file:
         pickle.dump(best_model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        #iterative_train_test_split method from the skmultilearn.model_selection 
        #is used to perform a stratified sampling of the multi-labeled data

        X_train, Y_train, X_test, Y_test = iterative_train_test_split(X, Y, test_size = 0.3)
        X_train = X_train.flatten()
        X_test = X_test.flatten()
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
