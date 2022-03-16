import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, sent_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    The class StartingVerbExtractor is used to detect if the starting
    token in the text(message) is a verb or not.

    The function starting_verb of this class uses the nltk package to
    implement this check. If the starting token is a verb, it returns
    True else returns False.
    """

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    """
    The function normalizes the text by converting it to lower case, it uses
    the nltk package to tokenize the text to word tokens, and lemmatize the tokens.

    Parameters:
    text (str) : the text which needs to be parsed into tokens

    Returns:
    list : the list of parsed tokens
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

#drop target categories which have one or less class value
target_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)
for col in target_cols:
    if len(list(df[col].value_counts().index)) <= 1:
        df.drop(columns=col, inplace=True)

# load model
model = joblib.load("../models/model_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # group the dataframe by genere and get a count of mesages per genere
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    # melt the dataframe to create a category colum for the different categories
    # get the count of messages per category, and display top 20 categories
    df_cat = df.melt(id_vars=['id', 'message', 'genre','original'], var_name=['category'], value_name='cat_message_count')
    d1 = df_cat.groupby('category')['cat_message_count'].sum().sort_values(ascending=False).reset_index()
    d1 = d1.head(20)
    
    
    cat_names = list(d1['category'].values)
    cat_count = list(d1['cat_message_count'].values)
    
    # create visuals
    graphs = [
        
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Caegories - Top 20 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
