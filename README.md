# Disaster-Response-Piplines
Analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages.

![app](/app/app.jpg)

## Acknowledgement

* UDACITY (DSND).
* FIGUREIGHT (Disaster Response Data).

## Motivation


In this project, I worked worked with a lot of exciting topics, like:
- ETL Pipeline (Pandas-Numpy ...etc).
- NLP (NLTK).
- ML Pipeline (SKLearn).
- Delpoyment (Flask).
- Relational Databases (SQLite).


## File Description


* app folder:
     - **run.py**: we app intialzaton.
* data folder:
     - **check_db_results.ipynb**: to check the db contents.
     - **disaster_categories.csv**: data to process.
     - **disaster_messages.csv**: data to process.
     - **DisasterResponse.db**: resulted db file (from ETL).
     - **process_data.py**: ETL pipeline py file.
* models folder:
     - **model.pkl**: classification model.
     - **train_classifier.py**: ML pipeline.
     - **disaster_messages.csv**: data to process.
* ML Piplene Notebook.
* ETL Piplene Notebook.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the libraries in requirments file.

```bash
pip install -r requirements.txt
```





## Project Components

There are three components on this project. 
### 1. ETL Pipeline
to run the ETL Pupeline you need  three command line arguments:
- disaster_messages.csv file.
- disaster_categories.csv file.
- The DataBaseName.
```bash
python -m process_data disaster_messages.csv disaster_categories.csv DisasterResponse
```

   - Loading the messages and categories datasets.
   - Merging the two datasets
   - Data Cleaning.
   - Storing the data in a SQLite database.

### 2. ML Pipeline
```bash
python -m train_classifier ../data/DisasterResponse.db model.pkl
```
   - Loading data from the SQLite database.
   - Splitting the dataset into training and test sets.
   - Building a text processing and machine learning pipeline.
   - Training and tuning a model using GridSearchCV.
   - Showing results on the test set.
   - Exporting the final model as a pickle file.


### 3. Flask Web App
- Run the following command in the app's directory to run your web app.
    `python run.py`

- Go to http://127.0.0.1:5000/


