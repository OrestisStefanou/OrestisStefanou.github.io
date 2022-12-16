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
As a software engineer who works at a company that provides a platform to invest money in stocks i read the book ‘How to make money in stocks’ by William O'Neil to understand what kind of data we should provide to our clients to help them make better decisions.The book lead me this website https://www.investors.com/ibd-data-tables/ that in my opinion provide really valuable data but since these data are in table format i created an api to provide more flexible querying to these data. This api can help investors discover stocks that will possibly strengthen their portfolio but also help them avoid stocks that will possibly weaken their portfolio as well. All credits for the information provided is given to Investors.com, consider to subscribe to their website or buy the author's book. I created this api for my own personal use and for educational purposes, I don't make any money from it.

## System high level architecture

<center><img src='./assets/img/posts/20221213/architecture.png'></center>


## Data fetching high level flowchart

<center><img src='./assets/img/posts/20221213/flowchart1.png'></center>


## How is database getting populated with data
At the moment the database is getting populated manually by me by running a script once a week to scrape the data from investors.com and store them in the database