# import libraries
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score,  precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import warnings
import pickle
import sys
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def load_data(db_file):
    """
    Load Data from the Database
    
    Arguments:
        database_filepath
        X (data featuers)
        Y (data labels (classes))
        classes
    """
    engine = create_engine(f'sqlite:///{db_file}')

        # Get table information
    inspector = inspect(engine)

    # get table name
    table_name = inspector.get_table_names()[0]

    # load DB file into data frame
    df = pd.read_sql(f'select * from {table_name}',con=engine)

    
    # X, Y
    X = df['message']
    Y = df.iloc[:,4:]
    
    # classes
    classes = list(Y.columns)

    return X, Y, classes

def tokenize(text):
    """
    convert text into clean tokens
    Args: text
    Return: list of tokens
    """
    # normalize text; remove all special charchters and convert all letters to lowercase 
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    
    # token messages
    words = word_tokenize(text)
    
    # removing stop words
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    # emmtization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Building the model
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    print the model scores
    """
    y_pred = model.predict(X_test)
    
    # Print accuracy score, precision score, recall score and f1_score for each categories
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))

def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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