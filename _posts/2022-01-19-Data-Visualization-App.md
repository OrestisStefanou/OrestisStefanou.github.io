---
layout: post
read_time: true
show_date: true
title: "Data Visualization web app"
date: 2022-01-19
img: posts/20220119/data-visualization.png
tags: [python, dash]
author: Stefanou Orestis
description: "A simple web app where you can visualize your data easy and fast"
---
A few weeks ago i read about [plotly-dash](https://plotly.com/dash/) and i decided to build this simple web app to visualize your data to learn how the library works

## [Visit the app](http://orestis-visualization-app.herokuapp.com/)

## How to use the app
1. Choose your data source from the dropdown menu
   * If it's a database source fill in the host,user,password and database fields and then press the connect button
   * If it's a csv or excel just upload the file

2. Write your sql query in the textbox and press 'Run query' button in case you want to filter your data
   * If your data source is a csv or excel file you can write Expressions in string form to filter data. 
3. To visualize the data shown in the datatable choose the type of plot you need from the dropdown menu,choose the column for x and y axis and press the 'plot' button