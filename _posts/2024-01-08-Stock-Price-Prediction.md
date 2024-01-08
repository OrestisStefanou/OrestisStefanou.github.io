---
layout: post
read_time: true
show_date: true
title: "Stock Price Prediction End to End"
date: 2024-01-08
img: posts/20240108/algorithmic-trading.png
tags: [investing, stock-market, machine-learning]
author: Stefanou Orestis
description: "An end to end machine learning project to predict and serve stock price predictions"
---
The goal of this project is to develop a machine learning model that will predict if a stock price will be higher or lower six months from the time we request a prediction. After the model development is done we will also create a microservice that will serve the prediction probabilities along with the prediction factors(shap values). The model will be stock agnostic meaning that it will be trained with data from multiple stocks so that we can have a single model that will give predictions for any stock.

## Data Gathering
The source of our data will be the Alpha Vantage API(https://www.alphavantage.co/documentation/). More specifically the data we will use are the ones below.
- Weekly adjusted time series
- Company Overview
- Income Statement
- Balance Sheet
- Cash Flow
- Treasury Yield
- Federal Funds(Interest Rate)

## Feature Engineering and Dataset Creation
Our dataset should have the format below

| Date | Feature 1 | Feature 2 | Feature 3 | Target |
|------|-----------|-----------|-----------|--------|
|      |           |           |           |        |

In our case the target is binary value 0/1 with 0 meaning that the stock price will be lower six months from now and 1 meaning higher. The features that we will use are the ones below.
- Date (This will only be used to split our dataset in training and test set)
- Stock Sector
- Interest Rate
- Treasury Yield
- Stock returns last 6 months
- Stock returns last 3 months
- Stock returns last 1 month
- Stock price volatility last 6 months
- Stock price volatility last 3 months
- Stock price volatility last 1 month
- Stock Sector performance last 6 months
- Stock Sector performance last 3 months
- Stock Sector performance last 1 month
- Capital expenditure arctan percentage change quarter over quarter
- Cash and Cash Equivalents arctan percentage change quarter over quarter
- Cashflow from financing arctan percentage change quarter over quarter
- Cashflow from investment arctan percentage change quarter over quarter
- Current net receivables arctan percentage change quarter over quarter
- Dividend payout arctan percentage change quarter over quarter
- EBITDA arctan percentage change quarter over quarter
- Gross profit arctan percentage change quarter over quarter
- Inventory arctan percentage change quarter over quarter
- Long term debt arctan percentage change quarter over quarter
- Net income arctan percentage change quaerter over quarter
- Net interest income arctan percentage change quaerter over quarter
- Operating cashflow arctan percentage change quarter over quarter
- Operating income arctan percentage change quarter over quarter
- Payments for repurchase of equity arctan percentage change quarter over quarter
- Net proceeds from long-term debt and capital securities issuance arctan percentage change quarter over quarter
- Property plant equipment arctan percentage change quarter over quarter
- Total assets arctan percentage change quarter over quarter
- Total current assets arctan percentage change quarter over quarter
- Total current liabilities arctan percentage change quarter over quarter
- Total liabilities arctan percentage change quarter over quarter
- Total revenue arctan percentage change quarter over quarter
- Total shareholder equity arctan percentage change quarter over quarter
- Current debt arctan percentage change quarter over quarter
- Cost of services sold arctan percentage change quarter over quarter
- Current debt arctan percentage change quarter over quarter
- Common stock shares outstanding arctan percentage change quarter over quarter
- Earnings per share
- Revenue per share
- Book value per share
- Gross profit margin
- Operating profit margin
- Return on assets
- Return on equity
- Cash to debt ratio
- Assets to liabilities ratio
- Price to earnings ratio
- Price to book ratio
- Price to sales ratio

The reason we are using arctan percentage change instead of plain percentage change is because we have cases like this one <br/>
Q1 total revenue = 0 <br/>
Q2 total revenue = 1000000 <br/>
In cases where any of the fields from the financial statements was zero we replaced it with a very small value(0.01) to be able to calculate the percentage change. In cases like this though the percentage change would be a huge number so we are using the arctan percentage for a better data distribution.

The data above are stored in an sqlite table with name `price_prediction_dataset`. By running the code below we split our dataset to train and test set.

```
import datetime as dt
import sqlite3

import numpy as np
import pandas as pd

def split_data_to_train_and_test(
    df: pd.DataFrame,
    cutoff_date: dt.datetime,
    cutoff_date_column_name: str
) -> Tuple[pd.DataFrame]:
    """
    Returns (train_set_df, test_set_df)
    """
    df['DateColumn'] = pd.to_datetime(df[cutoff_date_column_name])
    # Split the data into train and test based on the cutoff date
    train_set = df[df['DateColumn'] < cutoff_date].copy()
    test_set = df[df['DateColumn'] >= cutoff_date].copy()

    train_set.drop(['DateColumn',], axis=1, inplace=True)
    test_set.drop(['DateColumn',], axis=1, inplace=True)
    
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    return train_set, test_set 


db_conn = sqlite3.connect('database.db')

query = '''
    SELECT * 
    FROM price_prediction_dataset
    WHERE DATE(Date) <= date('now', '-6 months')
    ORDER BY DATE(Date)
'''

dataset = pd.read_sql(query, db_conn)
dataset.dropna(inplace=True)

# Create categorical target
bins = [-float('inf'), 0, float('inf')]
labels = ['down', 'up']
label_mapping = {0: 'down', 1: 'up'}

dataset['next_six_months_pct_change_range'] = pd.cut(
    dataset['price_pct_change_next_six_months'],
    bins=bins,
    labels=[0, 1],
    right=False
)

train_set, test_set = utils.split_data_to_train_and_test(
    df=dataset,
    cutoff_date=dt.datetime(2023,5,1),
    cutoff_date_column_name='Date'
)

cols_to_drop = ['symbol', 'Date', 'price_pct_change_next_six_months', 'next_six_months_pct_change_range']
target_col = 'next_six_months_pct_change_range'

y_train = train_set[target_col]
X_train = train_set.drop(cols_to_drop, axis=1)

y_test = test_set[target_col]
X_test = test_set.drop(cols_to_drop, axis=1)

```

## Modelling
We will try algorithms from different families to find which one performs the best and more specifically these ones:
- XGBoost
- Random Forrest
- KNN
- Support Vector Machine
- MLP


```
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import (
    OneHotEncoder,
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import pandas as pd


classifiers = {
    'RandomForest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'MLP': MLPClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}

column_transformer = make_column_transformer(
    (
        OneHotEncoder(), ['sector']
    ),
    remainder='passthrough'
)

# Loop through classifiers
for name, classifier in classifiers.items():
    classifier_pipeline = make_pipeline(column_transformer, classifier)
    classifier_pipeline.fit(X_train, y_train)
    y_pred = classifier_pipeline.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPerformance for {name}:")
    print(f"Overall Accuracy: {accuracy:.2%}")

    # Plot the confusion matrix
    y_test_labels = [label_mapping[y] for y in y_test]
    y_pred_labels = [label_mapping[y] for y in y_pred]
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=labels, normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
``` 

### Results
**Random Forest Classifier**
Overall Accuracy: 72.84% <br/> 
Normalized confusion matrix<br/>

| Label | Down | Up  |
| :---  | :--: | --: |
| Down  | 0.74 | 0.26|
| Up    | 0.28 | 0.72|


**XGBoost Classifier**
Overall Accuracy: 63.61% <br/> 
Normalized confusion matrix<br/>

| Label | Down | Up  |
| :---  | :--: | --: |
| Down  | 0.70 | 0.30|
| Up    | 0.41 | 0.59|


**KNN Classifier**
Overall Accuracy: 59.50% <br/> 
Normalized confusion matrix<br/>

| Label | Down | Up  |
| :---  | :--: | --: |
| Down  | 0.60 | 0.40|
| Up    | 0.41 | 0.59|


**MLP Classifier**
Overall Accuracy: 53.37% <br/> 
Normalized confusion matrix<br/>

| Label | Down | Up  |
| :---  | :--: | --: |
| Down  | 0.45 | 0.55|
| Up    | 0.36 | 0.64|


**SVC**
Overall Accuracy: 55.37% <br/> 
Normalized confusion matrix<br/>

| Label | Down | Up  |
| :---  | :--: | --: |
| Down  | 0.53 | 0.47|
| Up    | 0.38 | 0.62|
