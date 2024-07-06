---
layout: post
read_time: true
show_date: true
title: "Stock investor assistant chatbot"
date: 2024-07-06
img: posts/20240706/chatbot.png
tags: [gen-ai, stocks, investing]
author: Stefanou Orestis
description: "How to use langchain to create a stock investor assistant chatbot"
---

## Table of Contents
- [Project Goal](#project-goal)
- [Solution](#solution)

## Project Goal
The goal of the project is to build a question/answering system over an SQL database that contains various financial data. We will also expose this chatbot through a web api so that multiple users can use it.

## Solution
The solution will be broken into these steps
- Prompt engineering
- Creating an sql agent using Langchain
- Creating chat message history
- Exposing the agent through a web api
