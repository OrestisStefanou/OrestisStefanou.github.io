---
layout: post
read_time: true
show_date: true
title: "How to manage a pool of machine learning models for crypto trading"
date: 2024-06-03
img: posts/20240603/crypto_trading.jpg
tags: [crypto, ml-ops, machine-learning]
author: Stefanou Orestis
description: "How to use MLFlow to manage end to end machine learning workflows that will be used for crypto trading."
---

## Table of Contents
- [Project Goal](#project-goal)
- [Solution](#solution)
  - [Data Gathering](#data-gathering)
  - [Deployment Pipeline](#deployment-pipeline)


## Project Goal
The goal of the project is to be able to predict the price movement of the 20 most established cryptocurrencies in the next 10 days and more specifically if the 10 day moving average will be at least 5% higher(in this case we buy) or lower(in this case we sell) 10 days from the time of the prediction.

## Solution
For each cryptocurrency we will have 2 binary classifiers.
- Downtrend classifier: This classifier will predict if the 10day moving average will be at least 5% lower in 10 days than the 10day moving average at the time of the prediction.
- Uptrend classifier: This classifier will predict if the 10day moving average will be at least 5% higher in 10 days than the 10day moving average at the time of the prediction.

In the next sections we will cover the following:
- Data gathering and feature engineering
- Deployment Pipeline
- Batch Predictions
- Prediction serving through a FastAPI web service
- Using MLFlow to automate the machine learning workflows

### Data Gathering
The machine learning models will be trained with various technical indicators that we will get from Alpha Vantage(http://alphavantage.co/). The technical indicators were chosen mostly based on their popularity and are the ones below:
- OBV (on balance volume)
- AD (Chaikin A/D line)
- ADX (average directional movement index)
- AROON
- MACD (moving average convergence / divergence)
- RSI (relative strength index)
- STOCH (stochastic oscillator values)
- MFI (money flow index)
- DX (directional movement index)
- TRIX (1-day rate of change of a triple smooth exponential moving average )
- BBANDS (Bollinger bands)
- PPO (percentage price oscillator)

Below is the code that will be used to fetch the data from Alpha Vantage and to perform feature engineering to create new features from the features above.

```python
import pandas as pd
import httpx

import settings


class DataGenerator:
    def __init__(self, symbol: str, fetch_data: bool = True) -> None:
        self.symbol = symbol
        if fetch_data:
            self.data = self._fetch_data()
        else:
            self.data = None

    def _get_techninal_indicator_daily_time_series(
        self,
        indicator: str,
        time_period: int = None
    ) -> pd.DataFrame:
        if time_period is None:
            time_period = settings.time_period

        params = {
            'apikey': settings.apikey,
            'symbol': f"{self.symbol}USDT",
            'function': indicator,
            'interval': settings.interval,
        }

        if indicator == 'AD':
            indicator = 'Chaikin A/D'

        if indicator in ['SMA', 'WMA', 'RSI', 'BBANDS', 'TRIX']:
            params['time_period'] = time_period
            params['series_type'] = settings.series_type

        if indicator in ['DX', 'MFI', 'AROON', 'ADX']:
            params['time_period'] = time_period

        if indicator in ['MACD', 'PPO']:
            params['series_type'] = settings.series_type

        json_response = httpx.get(
            url='https://www.alphavantage.co/query',
            params=params
        ).json()
        time_series = []

        for date, data in json_response[f"Technical Analysis: {indicator}"].items():
            if indicator == 'STOCH':
                time_series.append(
                {
                    "date": date,
                    "SlowK": float(data["SlowK"]),
                    "SlowD": float(data["SlowD"])
                }
            )
            elif indicator == 'AROON':
                time_series.append(
                {
                    "date": date,
                    "AroonDown": float(data["Aroon Down"]),
                    "AroonUp": float(data["Aroon Up"])
                }
            )
            elif indicator == 'MACD':
                time_series.append(
                {
                    "date": date,
                    "MACD": float(data["MACD"]),
                    "MACD_Signal": float(data["MACD_Signal"]),
                    "MACD_Hist": float(data["MACD_Hist"]),
                }
            )
            elif indicator == 'BBANDS':
                time_series.append(
                {
                    "date": date,
                    "Real_Upper_Band": float(data["Real Upper Band"]),
                    "Real_Lower_Band": float(data["Real Lower Band"])
                }
                )
            else:
                if "time_period" in params:
                    indicator_name = f"{time_period}_day_{indicator}"
                else:
                    indicator_name = indicator
                time_series.append(
                    {
                        "date": date,
                        indicator_name: float(data[indicator])
                    }
                )

        return pd.DataFrame(time_series)

    def _get_crypto_daily_time_series(self, market: str = 'USD') -> pd.DataFrame:
        json_response = httpx.get(f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={self.symbol}&market={market}&apikey=KNPL6J9N740SLRRG').json()
        time_series = []

        for date, data in json_response["Time Series (Digital Currency Daily)"].items():
            time_series.append(
                {
                    "date": date,
                    "open": float(data["1. open"]),
                    "high": float(data["2. high"]),
                    "low": float(data["3. low"]),
                    "close": float(data["4. close"]),
                    "volume": float(data["5. volume"]),
                }
            )

        return pd.DataFrame(time_series)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data['OBV_pct_change'] = data['OBV'].pct_change() * 100
        data['AD_pct_change'] = data['Chaikin A/D'].pct_change() * 100
        data['7_day_TRIX'] = data['7_day_TRIX'] * 100
        data['BBANDS_distance_pct'] = ((data['Real_Upper_Band'] - data['Real_Lower_Band']) / data['Real_Lower_Band']) * 100
        data['2_day_SMA_10_day_SMA_pct_diff'] = ((data['2_day_SMA'] - data['10_day_SMA']) / data['10_day_SMA']) * 100
        data['2_day_SMA_20_day_SMA_pct_diff'] = ((data['2_day_SMA'] - data['20_day_SMA']) / data['20_day_SMA']) * 100
        data['10_day_SMA_20_day_SMA_pct_diff'] = ((data['10_day_SMA'] - data['20_day_SMA']) / data['20_day_SMA']) * 100
        data.drop(columns=['OBV', 'Chaikin A/D', 'Real_Upper_Band', 'Real_Lower_Band', '2_day_SMA', '10_day_SMA', '20_day_SMA', 'date'], axis=1, inplace=True)
        data.dropna(inplace=True)
        columns_to_convert = data.columns[data.columns != 'target']
        data[columns_to_convert] = data[columns_to_convert].astype(float)
        return data

    def _fetch_data(self) -> pd.DataFrame:
        time_series_df = self._get_techninal_indicator_daily_time_series(
            indicator="SMA",
            time_period=10
        )

        indicators_dfs = []
        indicators_dfs.append(self._get_techninal_indicator_daily_time_series(indicator="SMA", time_period=2))
        indicators_dfs.append(self._get_techninal_indicator_daily_time_series(indicator="SMA", time_period=20))
        for indicator in settings.indicators:
            indicators_dfs.append(self._get_techninal_indicator_daily_time_series(indicator))

        merged_df = pd.merge(time_series_df, indicators_dfs[0], on='date')
        for df in indicators_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='date')

        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df.sort_values(by='date', inplace=True)
        merged_df.reset_index(inplace=True, drop=True)
        return merged_df

    def get_dataset(
        self,
        look_ahead_days: int = settings.prediction_window_days,
        downtrend: bool = False
    ) -> pd.DataFrame:
        """
        Returns the dataset that will be user for training and evaluation of the models
        Params:
        - look_ahead_days: the prediction timeframe
        - downtrend: If True the target variable will contain 1 if the price will go down,
        If False the target variable will contain 1 if the price will go up
        """
        if self.data is None:
            data = self._fetch_data()
        else:
            data = self.data.copy()

        # Creata a new column with the target variable
        data['future_SMA'] = data['10_day_SMA'].shift(-look_ahead_days)
        data.dropna(inplace=True)
        percentage_difference = (data['future_SMA'] - data['10_day_SMA']) / data['10_day_SMA']
        if downtrend:
            data['target'] = (percentage_difference <= settings.target_downtrend_pct).astype(int)
        else:
            data['target'] = (percentage_difference >= settings.target_uptrend_pct).astype(int)

        data.drop(columns=['future_SMA',], axis=1, inplace=True)
        data = self._transform_data(data)
        data.dropna(inplace=True)
        
        return data

    def get_prediction_input(self, number_of_instances: int = 1) -> pd.DataFrame:
        if self.data is None:
            data = self._fetch_data()
        else:
            data = self.data.copy()

        data = self._transform_data(data)
        return data.tail(number_of_instances)
```

### Deployment Pipeline
The deployment pipeline will take as parameters the symbol of the cryptocurrency and the trend type(uptrend/downtrend). The pipeline is broken down into three tasks
1. Split the dataset into train and test set.
2. Train various machine learning models.
3. Find best performing model and if it passes the performance thresholds store it in model registry.

We will use MLFlow for experiment tracking and model deployment. A very high level of the deployment pipeline flow is shown in the flowchart below
<center><img src='./assets/img/posts/20240603/DeploymentPipelineFlowchart.drawio.png'></center>
