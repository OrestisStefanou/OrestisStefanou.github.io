---
layout: post
read_time: true
show_date: true
title: "Investor Discovery Service"
date: 2022-12-13
img: posts/20221213/Investing.jpg
tags: [investing, stock-market, python]
author: Stefanou Orestis
description: "A web service that provides market aggregation data"
---

Investor Discovery Service is a web api that provides market data to help any investor who wants to take data driven decisions. It provides data that help you compare the performance of a stock compared to the rest of the stocks, the financials of a stock and the state of the economy and the market in general. It also provides estimations of stocks prices in different timelines using machine learning.

## Provide market data on three levels
1. The Big Picture
    1. Major indices performance data
    2. Economic Indicators data
2. Sectors
    1. Performance of each sector
    2. Top performing stocks from each sector
3. Stocks
    1. Metrics that compare the performance of a stock with the rest of the stocks

## The Big Picture
- Major indices performance
    - S&P 500
    - Dow Jones Industrial Avg
    - NYSE Composite
    - Nasdaq Composite
- Economic Indicators
    - Inflation
    - Unemployment Rate
    - Treasury Yield
    - Interest Rates

## Sectors
- Historical performance data for each sector
- Top performing stocks from each sector

## Stock
- Metrics that help you compare the performance of a stock with the rest of the stocks
    - eps_rating -> Earnings per Share Rating Indicates a Company’s Relative Earnings Growth Rate.
    Strong earnings growth is essential to a stock’s success and has the greatest impact on its future price performance.
    The EPS rating calculates the growth and stability of each company’s earnings over the last three years, giving additional weight to the most recent few quarters. The result is compared with those of all other common stocks in the price tables and is rated on a scale from 1 to 99, with 99 being the best.
    Example: An EPS rating of 90 means it outperformed 90% of all stocks.

    - rs_rating -> Relative Price Strength Rating Shows Emerging Price Leaders
    The Relative Price Strength (RS) rating shows you which stocks are the best price performers, measuring a stock’s performance over the previous 12 months. That performance is then compared with the performance of all other publicly traded companies and given a 1 to 99 rating, with 99 being best.
    Example: An RS rating of 85 means the stock’s price movement has outperformed 85% of all other common stocks in the last year. The greatest winning stocks since 1952 and even much earlier showed an average RS rating of 87 when they broke out of their first price consolidation areas (bases). In other words, the greatest stocks were already outperforming nearly 90%, or nine out of 10, of all other stocks in the market before they made their biggest price gains.

    - smr_rating -> The Sales + Profit Margins + Return on Equity (SMR) rating combines these important fundamental factors and is the fastest way to identify truly outstanding companies with real sales growth and profitability. These are factors that are widely followed by the better analysts and portfolio managers. The SMR rating is on an A to E scale, with A and B being the best. In most cases, you want to avoid stocks with an SMR rating of D or E.
    Example: An SMR rating of A puts a stock in the top 20% of companies in terms of sales growth, profitability, and return on equity.

    - acc_dis_rating -> Accumulation/Distribution—The Influence of Professional Trading on Stocks. It tells you if your stock is under accumulation (professional buying) or distribution (professional selling).Stocks are rated on an A to E scale, with each letter representing the following:
    A = heavy accumulation (buying) by institutions
    B = moderate accumulation (buying) by institutions
    C = equal (or neutral) amount of buying and selling by institutions
    D = moderate distribution (selling) by institutions
    E = heavy distribution (selling) by institutions

    - overall_rating -> A combination of the 4 ratings above, because of the impact of earnings and previous price performance on stock price, double weighting is given to both the Earnings per Share and the Relative Price Strength ratings. Normal weight is given to the Industry Group Relative Strength, SMR, and Accumulation/Distribution ratings.

    - vol_chg_pct -> Volume traded yesterday vs average daily volume last 50 days.

- Price estimations using machine learning
  - Using the financials of a stock along with many other factors(sector, interest rates, treasury yields etc.) a machine learning model is trained that returns the probabilities of the price going up or down 3 months and 6 months from today. It also returns the factors that lead the model to these predictions so that the users can use it as an extra source of research instead of a black box.

## Stock Collections
- Tech Leaders: Stock leaders in the technology industry.
- REIT Leaders: Stock leaders in the real estate investment trust industry.
- Dividend Leaders: Stocks with high dividend yield and dividend growth %.
- Utility Leaders: Stock leaders in the utilities industry.
- Top 200 stocks:  Top 200 stocks with the highest overall rating. 
- EPS rating Leaders: Stocks with the highest eps rating.
- Price Relative Strength Leaders: Stocks with the highest price relative strength  rating.
- Stocks under heavy buying by institutions.
- Stock under heavy selling by institutions.


## Benefits for the users
- Can get a sense of the direction of the market.
- Identify highest and lowest performing industries.
- Find top performing stocks in each sector.
- Easy to understand metrics to compare a stock’s performance with the rest of the stocks.
- Different ways to filter stocks either by collections or sectors.


## Endpoints Documentation

### Economic Indicators
**GET /economic_indicators/{indicator}/time_series**
indicator:
- Treasury_Yield
- Interest_Rate
- Inflation
- Unemployment

Example response schema
```
{
  "indicator": "Treasury_Yield",
  "unit": "string",
  "time_series": [
    {
      "value": 0,
      "registered_date": "string",
      "registered_date_ts": 0
    }
  ]
}
```

### World Indices
**GET /world_indices/{index}/time_series**
index:
- S%26P%20500 -> S&P 500
- Dow%20Jones%20Industrial%20Average -> Dow Jones Industrial Average
- NASDAQ%20Composite -> NASDAQ Composite
- NYSE%20COMPOSITE -> NYSE Composite

Example response schema
```
{
  "index": "S&P 500",
  "time_series": [
    {
      "open_price": 0,
      "high_price": 0,
      "low_price": 0,
      "close_price": 0,
      "volume": 0,
      "registered_date": "string",
      "registered_date_ts": 0
    }
  ]
}
```

### Collections
**GET /collections/dividend_leaders**
Example response schema
```
[
  {
    "symbol": "string",
    "name": "string",
    "yield_pct": 0,
    "dividend_growth_pct": 0
  }
]
```

**GET /collections/reit_leaders**
Example response schema
```
[
  {
    "symbol": "string",
    "name": "string",
    "yield_pct": 0,
    "dividend_growth_pct": 0
  }
]
```

**GET /collections/utility_leaders**
Example response schema
```
[
  {
    "symbol": "string",
    "name": "string",
    "yield_pct": 0,
    "dividend_growth_pct": 0
  }
]
```

**GET /collections/tech_leaders**
Example response schema
```
[
  {
    "symbol": "string",
    "name": "string",
    "comp_rating": 0,
    "eps_rating": 0,
    "rs_rating": 0,
    "annual_eps_change_pct": 0,
    "last_qtr_eps_change_pct": 0,
    "next_qtr_eps_change_pct": 0,
    "last_qtr_sales_change_pct": 0,
    "return_on_equity": "string"
  }
]
```

**GET /collections/top_200_overall_rated_stocks**
Example response schema
```
[
  {
    "overall_rating": 0,
    "eps_rating": 0,
    "rs_rating": 0,
    "name": "string",
    "symbol": "string",
    "fifty_two_wk_high": 0,
    "closing_price": 0,
    "vol_chg_pct": 0,
    "acc_dis_rating": "string",
    "smr_rating": "string",
    "sector": "string"
  }
]
```

**GET /collections/eps_rating_leaders**
Example response schema
```
[
  {
    "overall_rating": 0,
    "eps_rating": 0,
    "rs_rating": 0,
    "name": "string",
    "symbol": "string",
    "fifty_two_wk_high": 0,
    "closing_price": 0,
    "vol_chg_pct": 0,
    "acc_dis_rating": "string",
    "smr_rating": "string",
    "sector": "string"
  }
]
```

**GET /collections/price_strength_rating_leaders**
Example response schema
```
[
  {
    "overall_rating": 0,
    "eps_rating": 0,
    "rs_rating": 0,
    "name": "string",
    "symbol": "string",
    "fifty_two_wk_high": 0,
    "closing_price": 0,
    "vol_chg_pct": 0,
    "acc_dis_rating": "string",
    "smr_rating": "string",
    "sector": "string"
  }
]
```

**GET /collections/stocks_under_heavy_buying**
Example response schema
```
[
  {
    "overall_rating": 0,
    "eps_rating": 0,
    "rs_rating": 0,
    "name": "string",
    "symbol": "string",
    "fifty_two_wk_high": 0,
    "closing_price": 0,
    "vol_chg_pct": 0,
    "acc_dis_rating": "string",
    "smr_rating": "string",
    "sector": "string"
  }
]
```

**GET /collections/stocks_under_heavy_selling**
Example response schema
```
[
  {
    "overall_rating": 0,
    "eps_rating": 0,
    "rs_rating": 0,
    "name": "string",
    "symbol": "string",
    "fifty_two_wk_high": 0,
    "closing_price": 0,
    "vol_chg_pct": 0,
    "acc_dis_rating": "string",
    "smr_rating": "string",
    "sector": "string"
  }
]
```

### Sectors
**GET /sectors/{sector}/stocks**

sector:
- ENERGY
- INSURANCE
- COMPUTER
- AGRICULTURE
- AEROSPACE
- METALS
- FOOD_BEV
- ELECTRONICS
- APPAREL
- OFFICE
- MACHINE
- RETAIL
- ALCOHOL_TOB
- CHEMICAL
- BUSINESS SVC
- MISC
- AUTO
- UTILITY
- S&Ls
- BANKS
- MEDICAL
- CONSUMER
- MINING
- TELECOM
- CHIPS
- MEDIA
- TRANSPORT
- BUILDING
- LEISURE
- REAL ESTATE
- SOFTWARE
- FINANCE
- INTERNET

Example response schema
```
[
  {
    "overall_rating": 0,
    "eps_rating": 0,
    "rs_rating": 0,
    "name": "string",
    "symbol": "string",
    "fifty_two_wk_high": 0,
    "closing_price": 0,
    "vol_chg_pct": 0,
    "acc_dis_rating": "string",
    "smr_rating": "string",
    "sector": "string"
  }
]
```

**GET /sectors/performance**
Example response schema
```
[
  {
    "date": "string",
    "sectors_performance": [
      {
        "sector": "ENERGY",
        "daily_price_change_pct": 0,
        "start_of_year_price_change_pct": 0
      }
    ]
  }
]
```

**GET /sectors/{sector}/performance**
Example response schema
```
[
  {
    "date": "string",
    "sector_performance": {
      "sector": "ENERGY",
      "daily_price_change_pct": 0,
      "start_of_year_price_change_pct": 0
    }
  }
]
```

### Stocks
**GET /stocks/{symbol}/profile**
Example response schema
```
{
  "overall_rating": 0,
  "eps_rating": 0,
  "rs_rating": 0,
  "name": "string",
  "symbol": "string",
  "fifty_two_wk_high": 0,
  "closing_price": 0,
  "vol_chg_pct": 0,
  "acc_dis_rating": "string",
  "smr_rating": "string",
  "sector": "string"
}
```

**GET /stocks/{symbol}/historical_performance**
Example response schema
```
{
  "symbol": "string",
  "name": "string",
  "sector": "ENERGY",
  "historical_performance": [
    {
      "date": "string",
      "performance": {
        "overall_rating": 0,
        "eps_rating": 0,
        "rs_rating": 0,
        "closing_price": 0,
        "vol_chg_pct": 0,
        "acc_dis_rating": "string",
        "smr_rating": "string"
      }
    }
  ]
}
```

### Machine Learning
**GET /price_predictions/fundamentals_models/six_months_prediction?symbol=MSFT**
Example response
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

**GET /price_predictions/fundamentals_models/three_months_prediction?symbol=MSFT**
Example response
```
{
  "prediction_probabilites": {
    "up": 0.40,
    "down": 0.60
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
