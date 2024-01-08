---
layout: post
read_time: true
show_date: true
title: "Stock Price Prediction End to End"
date: 2024-01-08
img: posts/20240108/algorithmic-trading.png
tags: [investing, stock-market, python]
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

The reason we are using arctan percentage change instead of plain percentage change is because we have cases like this one
Q1 total revenue = 0
Q2 total revenue = 1000000
In cases where any of the fields from the financial statements was zero we replaced it with a very small value(0.01) to be able to calculate the percentage change. In cases like this though the percentage change would be a huge number so we are using the arctan percentage for a better data distribution.