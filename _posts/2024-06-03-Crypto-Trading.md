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

# Table of Contents
- [Project Goal](#what-is-the-goal-of-the-project)
- [Solution](#solution)
  - [Data Gathering](#data-gathering)
  - [Deployment Pipeline](#deployment-pipeline)


## What is the goal of the project
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


### Deployment Pipeline