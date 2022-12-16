---
layout: post
read_time: true
show_date: true
title: "Investor API"
date: 2022-12-13
img: posts/20221213/Investing.jpg
tags: [investing, graphql, python]
author: Stefanou Orestis
description: "An API that provides market aggregation data"
---
As a software engineer who works at a company that provides a platform to invest money in stocks i read the book ‘How to make money in stocks’ by William O'Neil to understand what kind of data we should provide to our clients to help them make better decisions.The book lead me to this website https://www.investors.com/ibd-data-tables/ that in my opinion provides really valuable data but since these data are in table format i created an api to provide more flexible querying to these data. This api can help investors discover stocks that will possibly strengthen their portfolio but also help them avoid stocks that will possibly weaken their portfolio as well. All credits for the information provided is given to Investors.com, consider to subscribe to their website or buy the author's book. I created this api for my own personal use and for educational purposes, I don't make any money from it.

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
