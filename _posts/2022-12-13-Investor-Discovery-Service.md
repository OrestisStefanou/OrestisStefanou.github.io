---
layout: post
read_time: true
show_date: true
title: "Investor Discovery Service"
date: 2022-12-13
img: posts/20221213/Investing.jpg
tags: [investing, graphql, python]
author: Stefanou Orestis
description: "A web service that provides market aggregation data"
---
As a software engineer who works at a company that provides a platform to invest money in stocks i read the book ‘How to make money in stocks’ by William O'Neil to understand what kind of data we should provide to our clients to help them make better decisions.The book lead me to this website https://www.investors.com/ibd-data-tables/ that in my opinion provides really valuable data but since these data are in table format i created an api to provide more flexible querying to these data. This api can help investors discover stocks that will possibly strengthen their portfolio but also help them avoid stocks that will possibly weaken their portfolio as well. All credits for the information provided is given to Investors.com, consider to subscribe to their website or buy the author's book. I created this api for my own personal use and for educational purposes, I don't make any money from it.

## Source code
[Code](https://github.com/OrestisStefanou/InvestorAPI)

## System high level architecture

<center><img src='./assets/img/posts/20221213/architecture.png'></center>


## Data fetching high level flowchart

<center><img src='./assets/img/posts/20221213/flowchart1.png'></center>


## How is database getting populated with data
At the moment the database is getting populated manually by me by running a script once a week to scrape the data from investors.com and store them in the database

## GraphQL schema overview

### Root types
```
query: Query {
    topCompositeStocks(
        day: Int!
        month: Int!
        year: Int!
        limit: Int! = 200
    ): [CompositeStock!]!

    bottomCompositeStocks(
        day: Int!
        month: Int!
        year: Int!
        limit: Int! = 200
    ): [CompositeStock!]!

    stocksWithSector(
        day: Int!
        month: Int!
        year: Int!
        sector: String!
    ): [SectorStock!]!

    sectorsPerformance(
        day: Int!
        month: Int!
        year: Int!
    ): [SectorPerformance!]!

    techLeaders(
        day: Int!
        month: Int!
        year: Int!
    ): [TechLeaderStock!]!

    topLowPricedStocks(
        day: Int!
        month: Int!
        year: Int!
    ): [LowPricedStock!]!

    dividendLeaders(
        day: Int!
        month: Int!
        year: Int!
    ): [StockLeader!]!

    reitLeaders(
        day: Int!
        month: Int!
        year: Int!
    ): [StockLeader!]!

    utilityLeaders(
        day: Int!
        month: Int!
        year: Int!
    ): [StockLeader!]!

    smallMidCapLeadersIndex(
        day: Int!
        month: Int!
        year: Int!
    ): [LeadersIndexStock!]!

    largeMidCapLeadersIndex(
        day: Int!
        month: Int!
        year: Int!
    ): [LeadersIndexStock!]!

    appereancesCountPerStockInCollection(
        collection: Collection!
        minCount: Int! = 1
        limit: Int! = 100
    ): [StockAppereancesCount!]!

    searchSymbolInCollection(
        symbol: String!
        collection: Collection!
    ): [CompositeStockStockLeaderTechLeaderStockLeadersIndexStock!]!
}

```
### Object Types
```
type CompositeStock {
    compRating: Int!
    epsRating: Int!
    rsRating: Int!
    accDisRating: String!
    fiftyTwoWkHigh: Float
    name: String!
    symbol: String!
    closingPrice: Float
    priceChangePct: Float
    volChgPct: Float
    registeredDate: String
}

type SectorStock {
    compRating: Int!
    epsRating: Int!
    rsRating: Int!
    accDisRating: String!
    fiftyTwoWkHigh: Float
    name: String!
    symbol: String!
    closingPrice: Float
    priceChangePct: Float
    volChgPct: Float
    registeredDate: String
    smrRating: String!
    sectorName: String!
    sectorDailyPriceChangePct: String!
    sectorStartOfYearPriceChangePct: String!
}

type SectorPerformance {
    sectorName: String!
    dailyPriceChangePct: Float!
    startOfYearPriceChangePct: Float!
}

type TechLeaderStock {
    name: String!
    symbol: String!
    closingPrice: Float!
    compRating: Int!
    epsRating: Int!
    rsRating: Int!
    annualEpsChangePct: Float
    lastQtrEpsChangePct: Float
    nextQtrEpsChangePct: Float
    lastQtrSalesChangePct: Float
    returnOnEquity: String
    registeredDate: String
}

type LowPricedStock {
    compRating: Int!
    epsRating: Int!
    rsRating: Int!
    accDisRating: String!
    fiftyTwoWkHigh: Float
    name: String!
    symbol: String!
    closingPrice: Float
    priceChangePct: Float
    volChgPct: Float
    registeredDate: String
    yearHigh: Float!
}

type StockLeader {
    name: String!
    symbol: String!
    closingPrice: Float!
    yieldPct: Float!
    dividendGrowthPct: Float!
    registeredDate: String
}

type LeadersIndexStock {
    compRating: Int!
    rsRating: Int!
    stockName: String!
    stockSymbol: String!
    closingPrice: Float!
    registeredDate: String
}

type StockAppereancesCount {
    symbol: String!
    name: String!
    count: Int!
}
```

## Object type's fields explanation
- epsRating -> Earnings per Share Rating Indicates a Company’s Relative Earnings Growth Rate.
Strong earnings growth is essential to a stock’s success and has the greatest impact on its future price performance.
The EPS rating calculates the growth and stability of each company’s earnings over the last three years, giving additional weight to the most recent few quarters. The result is compared with those of all other common stocks in the price tables and is rated on a scale from 1 to 99, with 99 being the best.
Example: An EPS rating of 90 means that a company’s bottom-line earnings results over the short and the long term are in the top 10% of the roughly 10,000 stocks being measured.

- rsRating -> Relative Price Strength Rating Shows Emerging Price Leaders
The Relative Price Strength (RS) rating shows you which stocks are the best price performers, measuring a stock’s performance over the previous 12 months. That performance is then compared with the performance of all other publicly traded companies and given a 1 to 99 rating, with 99 being best.
Example: An RS rating of 85 means the stock’s price movement has outperformed 85% of all other common stocks in the last year. The greatest winning stocks since 1952 and even much earlier showed an average RS rating of 87 when they broke out of their first price consolidation areas (bases). In other words, the greatest stocks were already outperforming nearly 90%, or nine out of 10, of all other stocks in the market before they made their biggest price gains.

- smrRating -> The Sales + Profit Margins + Return on Equity (SMR®) rating combines these important fundamental factors and is the fastest way to identify truly outstanding companies with real sales growth and profitability. These are factors that are widely followed by the better analysts and portfolio man- agers. The SMR rating is on an A to E scale, with A and B being the best. In most cases, you want to avoid stocks with an SMR rating of D or E.
Example: An SMR rating of A puts a stock in the top 20% of companies in terms of sales growth, profitability, and return on equity.

- accDisRating -> Accumulation/Distribution—The Influence of Professional Trading on Stocks
It tells you if your stock is under accumulation (professional buying) or distribution (professional selling). This thoroughly tested, complex, and proprietary formula is highly accurate and is not based on simple up/down volume calculations. Stocks are rated on an A to E scale, with each letter representing the following:
A = heavy accumulation (buying) by institutions
B = moderate accumulation (buying) by institutions
C = equal (or neutral) amount of buying and selling by institutions
D = moderate distribution (selling) by institutions
E = heavy distribution (selling) by institutions

- compRating -> A combination of the 4 ratings above, because of the impact of earnings and previous price performance on stock price, double weighting is given to both the Earnings per Share and the Relative Price Strength ratings. This weighting may change somewhat in the future as we continue to improve our ratings. Normal weight is given to the Industry Group Relative Strength, SMR, and Accumulation/Distribution ratings.
The percent off the stock’s 52-week high is also used in the SmartSelect Composite Rating.
The results are then compared to the entire database, and a 1 to 99 rating (with 99 being best)

- volChgPct -> Volume traded yesterday vs average daily volume last 50 days. 