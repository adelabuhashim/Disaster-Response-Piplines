import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load all data files to return one  df.
    Args: - all dfs.
    Returns: - merged df.
    """
    # read in file
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories)
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row =  list(categories.iloc[:1].values[0])

    # get category column names
    category_colnames = [i[:-2] for i in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    def binary_conversion(text):
        """
        convert column_name_0 and column_name_1 to 0 and 1
        """
        return int(text[-1])
    
    for column in categories:
        categories[column] = categories[column].apply(binary_conversion)
     
    
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concat dfs
    df = pd.concat([df, categories], axis=1)

    return df



def clean_data(df):
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    save df into sqlite db
    Args: df to be saved, db file name
    Note: the table name would be the same db file name
    """
    engine = create_engine(f'sqlite:///{database_filename}.db')
    df.to_sql(f'{database_filename}', engine, index=False,  if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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