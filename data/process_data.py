#import the libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    The function loads the messages and categories datasets
    from the csv files to pandas dataframes.
    
    Parameters:
    messages_filepath (str) : path to the messages csv file
    categories_filepath (str) : path to the categories csv file
    
    Returns: 
    DataFrame : merged dataframe contianing data from messages and categories datasets 
    """
    
    #load the messages and categories datasets
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    categories = pd.read_csv(categories_filepath)

    #merge the datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    The function modifies the dataframe to create individual columns for the categories,
    convert the category values to numeric values 0 or 1, and drop duplicates entries.  
    
    Parameters:
    df (DataFrame) : the original dataframe

    Returns:
    DataFrame : the modified cleaned dataframe
    """

    #create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand=True)

    #select the first row of the categories dataframe
    row = categories.iloc[0]

    #use this row to extract a list of new column names for categories
    category_colnames = list(pd.Series(row.values).apply(lambda x : x[0:-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #convert category values to just number 0 or 1
    for column in categories:
        #set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        # if the value is 2.0, set it to 1.0
        categories.loc[categories[column] == 2.0, column] = 1.0

    #drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)    

    #concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(subset=['id', 'message', 'original', 'genre'], keep='first', inplace=True)
    return df

def save_data(df, database_filename):
    """
    The function saves the dataframe as a table to a sqlite database.

    Parameters:
    df (DataFrame) : dataframe which needs to be saved as a table

    Returns:
    str : path to the database where the dataframe needs to be saved
    """
    db_connect_str = 'sqlite:///{}'.format(database_filename)
    engine = create_engine(db_connect_str)
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.head())
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
