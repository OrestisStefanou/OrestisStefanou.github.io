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


From the results above we can see that Random Forest outperforms the rest of the algorithms so we choose this one for our predictions.
### Feature Importance (Top 20)

| Feature                                            | Importance          |
| -------------------------------------------------- | -------------------- |
| treasury_yield                                     | 0.069806             |
| interest_rate                                      | 0.038903             |
| sector_pct_change_last_six_months                  | 0.038497             |
| sector_pct_change_last_month                       | 0.030899             |
| sector_pct_change_last_three_months                | 0.029667             |
| price_volatility_last_six_months                   | 0.027291             |
| price_volatility_last_three_months                 | 0.025358             |
| price_to_sales_ratio                               | 0.024609             |
| price_to_book_ratio                                | 0.023293             |
| price_volatility_last_month                        | 0.022188             |
| price_pct_change_last_six_months                   | 0.022102             |
| price_pct_change_last_month                        | 0.022063             |
| pe_ratio                                           | 0.021414             |
| price_pct_change_last_three_months                 | 0.021333             |
| return_on_assets                                   | 0.019453             |
| operating_profit_margin                            | 0.019330             |
| cash_to_debt_ratio                                 | 0.019183             |
| assets_to_liabilities_ratio                        | 0.018904             |
| property_plant_equipment_arctan_pct_change         | 0.018851             |
| revenue_per_share                                  | 0.018741             |


## Store the model
We store the model to use it later in our microservice

```
import joblib

joblib.dump(rf_six_months_classifier, 'rf_six_months_prediction_model.joblib')
```

# Microservice development
Now that we have our random forest stored we will develop a web microservice using FastAPI so that we can serve the prediction probabilities along with the prediction factors(shap values) to anyone.

## Predictor Class
The first step is to create a wrapper on top of our machine learning model.

```
import datetime as dt
from typing import Dict, List, Set, Any, Optional
import joblib

import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from analytics.utils import (
    get_stock_time_series_df,
    get_stock_symbols
)
from analytics.machine_learning.price_prediction_with_fundamentals.utils import (
    get_stock_fundamental_df,
    add_timeseries_features,
    get_sector_time_series
)
from analytics.errors import (
    PredictionDataNotFound,
    InvalidPredictionInput
)

features_map = {
    'onehotencoder__sector_ENERGY & TRANSPORTATION': 'Stock Sector',
    'onehotencoder__sector_FINANCE': 'Stock Sector',
    'onehotencoder__sector_LIFE SCIENCES': 'Stock Sector',
    'onehotencoder__sector_MANUFACTURING': 'Stock Sector',
    'onehotencoder__sector_REAL ESTATE & CONSTRUCTION': 'Stock Sector',
    'onehotencoder__sector_TECHNOLOGY': 'Stock Sector',
    'onehotencoder__sector_TRADE & SERVICES': 'Stock Sector',
    'remainder__interest_rate': 'Interest Rates', 
    'remainder__treasury_yield': 'Treasury Yield',
    'remainder__price_pct_change_last_six_months': 'Stock returns last 6 months',
    'remainder__price_pct_change_last_three_months': 'Stock returns last 3 months',
    'remainder__price_pct_change_last_month': 'Stock returns last 1 month',
    'remainder__price_volatility_last_six_months': 'Stock price volatility last 6 months',
    'remainder__price_volatility_last_three_months': 'Stock price volatility last 3 months',
    'remainder__price_volatility_last_month': 'Stock price volatility last 1 month',
    'remainder__sector_pct_change_last_six_months': 'Stock Sector performance last 6 months',
    'remainder__sector_pct_change_last_three_months': 'Stock Sector performance last 3 months',
    'remainder__sector_pct_change_last_month': 'Stock Sector performance last 1 month',
    'remainder__capital_expenditures_arctan_pct_change': 'Capital expenditure change quarter over quarter',
    'remainder__cash_and_cash_equivalents_at_carrying_value_arctan_pct_change': 'Cash and Cash Equivalents change quarter over quarter',
    'remainder__cashflow_from_financing_arctan_pct_change': 'Cashflow from financing change quarter over quarter',
    'remainder__cashflow_from_investment_arctan_pct_change': 'Cashflow from investment change quarter over quarter',
    'remainder__current_net_receivables_arctan_pct_change': 'Current net receivables change quarter over quarter',
    'remainder__dividend_payout_arctan_pct_change': 'Dividend payout change quarter over quarter',
    'remainder__ebitda_arctan_pct_change': 'EBITDA change quarter over quarter',
    'remainder__gross_profit_arctan_pct_change': 'Gross profit change quarter over quarter',
    'remainder__inventory_arctan_pct_change': 'Inventory change quarter over quarter',
    'remainder__long_term_debt_arctan_pct_change': 'Long term debt change quarter over quarter',
    'remainder__net_income_arctan_pct_change': 'Net income change quaerter over quarter',
    'remainder__net_interest_income_arctan_pct_change': 'Net interest income change quaerter over quarter',
    'remainder__operating_cashflow_arctan_pct_change': 'Operating cashflow change quarter over quarter',
    'remainder__operating_income_arctan_pct_change': 'Operating income change quarter over quarter',
    'remainder__payments_for_repurchase_of_equity_arctan_pct_change': 'Payments for repurchase of equity change quarter over quarter',
    'remainder__proceeds_from_issuance_of_long_term_debt_and_capital_securities_net_arctan_pct_change': 'Net proceeds from long-term debt and capital securities issuance change quarter over quarter',
    'remainder__property_plant_equipment_arctan_pct_change': 'Property plant equipment change quarter over quarter',
    'remainder__total_assets_arctan_pct_change': 'Total assets change quarter over quarter',
    'remainder__total_current_assets_arctan_pct_change': 'Total current assets change quarter over quarter',
    'remainder__total_current_liabilities_arctan_pct_change': 'Total current liabilities change quarter over quarter',
    'remainder__total_liabilities_arctan_pct_change': 'Total liabilities change quarter over quarter',
    'remainder__total_revenue_arctan_pct_change': 'Total revenue change quarter over quarter',
    'remainder__total_shareholder_equity_arctan_pct_change': 'Total shareholder equity change quarter over quarter',
    'remainder__current_debt_arctan_pct_change': 'Current debt change quarter over quarter',
    'remainder__cost_of_goods_and_services_sold_arctan_pct_change': 'Cost of services sold change quarter over quarter',
    'remainder__current_debt_arctan_pct_change': 'Current debt change quarter over quarter',
    'remainder__common_stock_shares_outstanding_arctan_pct_change': 'Common stock shares outstanding change quarter over quarter',
    'remainder__eps': 'Earnings per share',
    'remainder__revenue_per_share': 'Revenue per share',
    'remainder__book_value_per_share': 'Book value per share',
    'remainder__gross_profit_margin': 'Gross profit margin',
    'remainder__operating_profit_margin': 'Operating profit margin',
    'remainder__return_on_assets': 'Return on assets',
    'remainder__return_on_equity': 'Return on equity',
    'remainder__cash_to_debt_ratio': 'Cash to debt ratio',
    'remainder__assets_to_liabilities_ratio': 'Assets to liabilities ratio',
    'remainder__pe_ratio': 'Price to earnings ratio',
    'remainder__price_to_book_ratio': 'Price to book ratio',
    'remainder__price_to_sales_ratio': 'Price to sales ratio',
}

class SixMonthsPriceMovementPredictor:
    """
    This class is used to predict if the price of a
    given stock will go up or down. It also returns the
    factors(shap values) that lead to the model's prediction.
    The _ml_model class variable is a Pipeline object 
    with two steps:
        1. Column transformer
        2. Classifier
    """
    _ml_model: Pipeline = joblib.load('rf_six_months_prediction_model.joblib')

    @classmethod
    def get_prediction_probabilities_with_prediction_factors(cls, symbol: str) -> Dict[str, Any]:
        """
        Returns a dictionary with the predicted probabilities for the 
        price of the symbol's stock. Example:
        {
            "prediction_probabilities": {
                "up": 0.75,
                "down": 0.25
            },
            "prediction_factors": {
                "up": ['Stock Sector', 'Net interest income change quaerter over quarter',],
                "down": ['Interest Rates', 'Total liabilities change quarter over quarter']
            }
        }
        """
        prediction_input = cls._create_stock_prediction_input_data(symbol)
        cls._validate_prediction_input(prediction_input)
        prediction_probabilites = cls._ml_model.predict_proba(prediction_input)
        down_prediction_factors = cls._get_prediction_factors(prediction_input, 0)
        up_prediction_factors = cls._get_prediction_factors(prediction_input, 1)
        return {
            'prediction_probabilities':{
                'down': prediction_probabilites[0][0],
                'up': prediction_probabilites[0][1]
            },
            'prediction_factors': {
                'down': list(down_prediction_factors),
                'up': list(up_prediction_factors)
            }
        }

    @classmethod
    def _get_prediction_factors(cls, prediction_input: pd.DataFrame, predicted_class: int) -> Set[str]:
        classifier = cls._ml_model.steps[1][1]
        explainer = shap.TreeExplainer(classifier)
        prediction_input_transformer = cls._ml_model.steps[0][1]
        cls._validate_prediction_input(prediction_input)
        shap_values = explainer.shap_values(prediction_input_transformer.transform(prediction_input))
        features = prediction_input_transformer.get_feature_names_out()
        features_with_shap_values = list()

        for i in range(len(features)):
            feature_name = features[i]
            shap_value = shap_values[predicted_class][0][i]
            if shap_value > 0:
                features_with_shap_values.append((feature_name, shap_value))
        
        return {
            features_map[x[0]] 
            for x in sorted(features_with_shap_values, key=lambda x: x[1], reverse=True)
        }

    @classmethod
    def _validate_prediction_input(cls, prediction_input: pd.DataFrame) -> None:
        if prediction_input is None:
            raise PredictionDataNotFound('Prediction data not available')

        nan_columns = prediction_input.columns[prediction_input.isna().any()].tolist()
        if len(nan_columns) > 0:
            raise InvalidPredictionInput(f"Features with NaN values: {nan_columns}")

    @classmethod
    def _create_stock_prediction_input_data(cls, symbol: str) -> Optional[pd.DataFrame]:
        stock_fundamental_df = get_stock_fundamental_df(symbol)
        if stock_fundamental_df.empty:
            return None

        stock_time_series_df = get_stock_time_series_df(symbol)

        stock_sector = stock_fundamental_df.iloc[0, stock_fundamental_df.columns.get_loc('sector')]
        sector_time_series_df = get_sector_time_series(stock_sector)

        if sector_time_series_df is None:
            return None
        
        # Create stock prediction data
        current_date = dt.datetime.today()
        stock_prediction_data_df = pd.DataFrame([
            {"Date": current_date},
        ])
        stock_prediction_data_df['symbol'] = symbol
        stock_prediction_data_df['sector'] = stock_sector

        return add_timeseries_features(
            stock_prediction_data_df=stock_prediction_data_df,
            stock_fundamental_df=stock_fundamental_df,
            stock_time_series_df=stock_time_series_df,
            sector_time_series_df=sector_time_series_df
        )

```

There is a lot happening above so let's break it down
- `_create_stock_prediction_input_data` -> This function given a symbol creates the input the machine learning model expects.
- `_validate_prediction_input` -> Checks if the input created by `_create_stock_prediction_input_data ` is valid and if not it raises an exception
- `_get_prediction_factors` -> This is the function that returns the shap values for a single prediction using the shap library. Since the feature names are not readable as they are we have the  `features_map` dictionary that maps a feature name with a more readable interpretation. For more regarding the shap values have a look here https://shap.readthedocs.io/en/latest/.
- `get_prediction_probabilities_with_prediction_factors` -> This is the function where it all comes together. Given a symbol this function returns a dictionary with the prediction probabilities and the prediction factors and is the function that the users of this class will use.


## FastAPI
```
from http import HTTPStatus

from fastapi import FastAPI
from fastapi import (
    APIRouter,
    HTTPException
)
import pydantic

import SixMonthsPriceMovementPredictor
from errors import (
    PredictionDataNotFound,
    InvalidPredictionInput
)

app = FastAPI()
router = APIRouter(prefix='/price_predictions')
app.include_router(router)


class PredictionProbabilities(pydantic.BaseModel):
    up: float
    down: float


class PredictionFactors(pydantic.BaseModel):
    up: List[str]
    down: List[str]


class FundamentalsPricePrediction(pydantic.BaseModel):
    prediction_probabilites: PredictionProbabilities
    prediction_factors: PredictionFactors


@router.get(
    "/fundamentals_models/six_months_prediction",
    tags=["Machine Learning"],
    status_code=200,
    response_model=FundamentalsPricePrediction
)
async def get_six_months_price_prediction_with_fundamentals(symbol: str):
    try:
        predictions_with_factors = SixMonthsPriceMovementPredictor.get_prediction_probabilities_with_prediction_factors(
            symbol=symbol
        )
    except PredictionDataNotFound:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail='Invalid or unsupported symbol'
        )
    except InvalidPredictionInput:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail='Something went wrong, we are working on it.'
        )

    # Serialize the response
    return FundamentalsPricePrediction(
        prediction_probabilites=PredictionProbabilities(
            up=predictions_with_factors['prediction_probabilities']['up'],
            down=predictions_with_factors['prediction_probabilities']['down']
        ),
        prediction_factors=PredictionFactors(
            up=predictions_with_factors['prediction_factors']['up'],
            down=predictions_with_factors['prediction_factors']['down'],
        )
    )

```

## Getting predictions from the api
After we start the server we can get predictions for Microsoft by calling the endpoint
```
curl -X 'GET' \
  'http://127.0.0.1:8000/price_predictions/fundamentals_models/six_months_prediction?symbol=MSFT' \
  -H 'accept: application/json'
```

Response
```
{
  "prediction_probabilites": {
    "up": 0.63,
    "down": 0.37
  },
  "prediction_factors": {
    "up": [
      "Total revenue change quarter over quarter",
      "Cost of services sold change quarter over quarter",
      "Common stock shares outstanding change quarter over quarter",
      "Operating cashflow change quarter over quarter",
      "Total current liabilities change quarter over quarter",
      "Operating income change quarter over quarter",
      "Net proceeds from long-term debt and capital securities issuance change quarter over quarter",
      "Earnings per share",
      "Capital expenditure change quarter over quarter",
      "Price to earnings ratio",
      "Dividend payout change quarter over quarter",
      "Return on equity",
      "Total assets change quarter over quarter",
      "Assets to liabilities ratio",
      "Cash and Cash Equivalents change quarter over quarter",
      "Total shareholder equity change quarter over quarter",
      "Price to book ratio",
      "Long term debt change quarter over quarter",
      "Stock returns last 6 months",
      "Stock Sector",
      "Cashflow from financing change quarter over quarter",
      "Treasury Yield",
      "Payments for repurchase of equity change quarter over quarter",
      "Total current assets change quarter over quarter",
      "Stock Sector performance last 3 months",
      "Cash to debt ratio",
      "Property plant equipment change quarter over quarter",
      "Return on assets",
      "Revenue per share",
      "Interest Rates"
    ],
    "down": [
      "Net interest income change quaerter over quarter",
      "Cashflow from investment change quarter over quarter",
      "Stock price volatility last 1 month",
      "Stock Sector performance last 6 months",
      "Gross profit margin",
      "Stock Sector performance last 1 month",
      "Stock price volatility last 6 months",
      "Current debt change quarter over quarter",
      "Stock returns last 3 months",
      "Total liabilities change quarter over quarter",
      "Net income change quaerter over quarter",
      "Operating profit margin",
      "Stock price volatility last 3 months",
      "Gross profit change quarter over quarter",
      "Stock Sector",
      "EBITDA change quarter over quarter",
      "Price to sales ratio",
      "Stock returns last 1 month",
      "Inventory change quarter over quarter",
      "Book value per share",
      "Current net receivables change quarter over quarter"
    ]
  }
}
```
