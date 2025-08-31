---
layout: post
read_time: true
show_date: true
title: "AI Investor Assistant API"
date: 2025-07-20
img: posts/20250720/AI-financial-advisor-cover-image.png
tags: [investing, stock-market]
author: Stefanou Orestis
description: "The AI Investor Assistant API is a plug-and-play SaaS backend that empowers any financial product"
---

# AI Investor Assistant API

## 🌟 Overview

**The AI Investor Assistant API is a plug-and-play backend** that empowers any financial product—be it an investment app, trading platform, portfolio tracker, or finance education tool—to offer intelligent, **personalized investor guidance** via natural language chat.

## Key Benefits

* 🚀 **Plug-and-Play AI Assistant** – No need to build AI models or pipelines; simply integrate and provide immediate value.
* 🧠 **Personalized Investor Insights** – Responses can be tailored based on user context, such as portfolio composition, risk profile, and interests.
* 🤖 **Advanced AI Models** – Behind the scenes, the service leverages **OpenAI** and **Google Gemini** (configurable per use case) for high-quality answers.
* 📊 **Trusted Market Data Sources** – All responses are generated using **institutional-grade financial data** for accuracy and reliability.
* 🔍 **Continuous Quality Evaluation** – All questions and responses are securely stored to monitor performance and improve answer quality over time.
* 💡 **Increased User Engagement & Retention** – Real-time, interactive investment guidance keeps users engaged and returning.
* ⏱ **Time & Cost Savings** – Avoid the cost of developing, training, and maintaining large language models and financial knowledge bases.
* 📚 **Enhanced Support Capabilities** – AI can handle FAQs, generate follow-up questions, and guide users toward actionable insights.

---

## Example End-to-End Flow

Below is a quick way to test the API using simple HTTP requests. You can use **curl**, **Postman**, or any HTTP client.

### Step 1 – Create a Session

```sh
POST /session
```

**Response:**

```json
{ "session_id": "abc123xyz" }
```

---

### Step 2 – Create a User (Optional)

```sh
POST /user_context
Content-Type: application/json

{
  "user_id": "user_123",
  "user_profile": {
    "risk_tolerance": "moderate",
    "investment_horizon": "5 years"
  },
  "user_portfolio": [
    { "asset_class": "stock", "symbol": "AAPL", "quantity": 15, "portfolio_percentage": 50 },
    { "asset_class": "etf", "symbol": "SPY", "quantity": 5, "portfolio_percentage": 50 }
  ]
}
```

---

### Step 3 – Extract Topic & Tags (Optional)

```sh
POST /chat/extract_topic_and_tags
Content-Type: application/json

{
  "question": "How did Apple perform last quarter?",
  "session_id": "abc123xyz"
}
```

**Example Response:**

```json
{
  "topic": "stock_overview",
  "topic_tags": {
    "stock_symbols": ["AAPL"]
  }
}
```

---

### Step 4 – Generate a Chat Response

```sh
POST /chat
Content-Type: application/json

{
  "question": "How did Apple perform last quarter?",
  "topic": "stock_overview",
  "session_id": "abc123xyz",
  "topic_tags": {
    "stock_symbols": ["AAPL"]
  }
}
```

Response will stream back the AI-generated answer.

---

### Step 5 – Generate Follow-Up Questions

```sh
POST /follow_up_questions
Content-Type: application/json

{
  "session_id": "abc123xyz",
  "number_of_questions": 3
}
```

**Example Response:**

```json
{
  "follow_up_questions": [
    "Would you like to see a breakdown of Apple's revenue?",
    "Should I compare Apple’s results to Microsoft?",
    "Do you want to explore Apple’s future growth potential?"
  ]
}
```

---

# Api Reference
# Session API

## Endpoint

### POST `/session`

Creates a new session and returns the session ID.

## Request Parameters

_No parameters are required in the request body or query._

## Response

### Success Response (201 Created)

#### Example Response Body:
```json
{
  "session_id": "abc123xyz"
}
```

### Error Response (500 Internal Server Error)

#### Example Response Body:
```json
{
  "error": "failed to create session"
}
```

## Notes
- This endpoint is used to create a new session on the server.
- A successful request returns a `session_id`, which can be used in the chat endpoint.
- If an error occurs during session creation, a relevant error message will be returned in the response body.

## Example Request
```sh
POST /session
```

This request would create a new session and return the newly generated session ID.

--- 

### GET `/session/:session_id`

Retrieves the conversation history for a given session ID.

## Path Parameter

| Parameter    | Type   | Required | Description                        |
| ------------ | ------ | -------- | ---------------------------------- |
| `session_id` | string | Yes      | Unique identifier for the session. |

## Response

### Success Response (200 OK)

```json
{
  "conversation": [
    {
      "actor": "user",
      "message": "Hi, I need help with my portfolio."
    },
    {
      "actor": "assistant",
      "message": "Sure! Can you tell me more about your goals?"
    }
  ]
}
```

#### Response Fields

| Field          | Type   | Description                                                      |
| -------------- | ------ | ---------------------------------------------------------------- |
| `conversation` | array  | Array of message objects in the session.                         |
| `actor`        | string | Sender of the message. Possible values: `"user"`, `"assistant"`. |
| `message`      | string | Text content of the message.                                     |

### Error Responses

#### 400 Bad Request

```json
{
  "error": "session with id abc123 not found"
}
```

#### 500 Internal Server Error

```json
{
  "error": "internal server error"
}
```

---


# Chat Completion API

## Endpoint

### POST `/chat`

Generates a streaming chat response based on the user's question, topic, session, and contextual tags (like stock or financial statement info).

## Request Body

| Field             | Type      | Required | Description                                                                 |
|------------------|-----------|----------|-----------------------------------------------------------------------------|
| `question`        | string    | Yes      | The user's question to be answered.                                         |
| `topic`           | string    | Yes      | The context/topic for the chat (e.g., "education", "markets").                |
| `session_id`      | string    | Yes      | A valid session ID created via the `/session` endpoint.                     |
| `topic_tags`      | object    | No       | Optional tags to add financial context. See `Topic Tags` below.             |

### Topic Tags (Optional `topic_tags` object)

| Field              | Type    | Required | Description                                                              |
|-------------------|---------|----------|--------------------------------------------------------------------------|
| `sector_name`      | string  | No       | Name of the relevant sector (e.g., "Technology").                         |
| `industry_name`    | string  | No       | Name of the industry (e.g., "Semiconductors").                            |
| `stock_symbols`    | string[]| No       | List of stock symbols (e.g., ["AAPL", "MSFT"]).                           |
| `balance_sheet`    | boolean | No       | Whether to include balance sheet context.                                |
| `income_statement` | boolean | No       | Whether to include income statement context.                             |
| `cash_flow`        | boolean | No       | Whether to include cash flow context.                                    |
| `etf_symbols`      | string[]| No       | List of ETF symbols.                                                     |
| `user_id`       | string  | No       | ID of user asking the question.(look at user context section below)                          |

### Example Request Body
```json
{
  "question": "How did Apple perform last quarter?",
  "topic": "stock_overview",
  "session_id": "abc123xyz",
  "topic_tags": {
    "stock_symbols": ["AAPL"]
  }
}
```

## Response

### Success Response (200 OK – Streamed)

The response is a stream of JSON-encoded text chunks representing the chat reply. Each chunk is a string:
```json
"Apple reported strong earnings with increased revenue in Q4..."
```

> Note: This is streamed using server-sent events (chunked HTTP), not returned as a complete JSON object.

### Error Responses

#### 400 Bad Request
Occurs when the request payload is invalid or missing required fields.

```json
{
  "error": "question field is required"
}
```

```json
{
  "error": "session not found"
}
```

```json
{
  "error": "invalid topic"
}
```

#### 500 Internal Server Error

```json
{
  "error": "an unexpected error occurred"
}
```

## Notes
- Fields `question`, `topic`, and `session_id` are required.
- The topic field available values can be retrieved using the `GET /topics` endpoint
- If `session_id` is invalid or expired, a 400 error will be returned.
- This endpoint returns a **streaming** response, suitable for chat UIs that render text incrementally.
- The `topic_tags` object allows for fine-grained control over the context of the AI's response, especially when discussing financials.
- You can use `POST /chat/extract_topic_and_tags` endpoint to get the topic and tags if you don't know them before hand.

## Example Request
```sh
POST /chat
Content-Type: application/json

{
  "question": "What is the sentiment of the latest market news?",
  "topic": "news",
  "session_id": "abc123xyz",
  "topic_tags": {}
}
```

This request would trigger a streamed AI response about the semiconductor industry.

---

## Endpoint

### POST `/chat/extract_topic_and_tags`

Extracts the main topic and relevant financial context tags (sector, industry, stock symbols, etc.) from a user's question. This is typically used as a preprocessing step before generating a chat response. You can check [this](topic_tag_extractor.md) for more details on how this works behind 
the scenes.

## Request Body

| Field        | Type   | Required | Description                                               |
| ------------ | ------ | -------- | --------------------------------------------------------- |
| `question`   | string | Yes      | The user's question from which to extract topic and tags. |
| `session_id` | string | Yes      | A valid session ID created via the `/session` endpoint.   |
| `user_id`    | string | No       | ID of the user asking the question.                       |


### Example Request Body

```json
{
  "question": "Tell me about the performance of Apple and Microsoft in the tech sector.",
  "session_id": "abc123xyz",
  "user_id": "some_user_id"
}
```

## Response

### Success Response (200 OK)

Returns the inferred topic and relevant financial tags extracted from the question.

| Field        | Type   | Description                                    |
| ------------ | ------ | ---------------------------------------------- |
| `topic`      | string | The identified topic of the question.          |
| `topic_tags` | object | Structured metadata tags related to the topic. |

#### Topic Tags (`topic_tags` object)

| Field              | Type      | Description                                          |
| ------------------ | --------- | ---------------------------------------------------- |
| `sector_name`      | string    | Name of the relevant sector (e.g., "Technology").    |
| `industry_name`    | string    | Name of the industry (e.g., "Consumer Electronics"). |
| `stock_symbols`    | string\[] | List of stock symbols involved in the question.      |
| `balance_sheet`    | boolean   | Whether the question involves balance sheet data.    |
| `income_statement` | boolean   | Whether the question involves income statement data. |
| `cash_flow`        | boolean   | Whether the question involves cash flow data.        |
| `etf_symbols`      | string\[] | List of etf symbols involved in the question         |
| `user_id`          | string    | user_id given in the request                         |

### Example Success Response

```json
{
  "topic": "stock_overview",
  "topic_tags": {
    "sector_name": "",
    "industry_name": "",
    "stock_symbols": ["AAPL", "MSFT"],
    "balance_sheet": false,
    "income_statement": false,
    "cash_flow": false,
    "etf_symbols": []
  }
}
```

### Error Responses

#### 400 Bad Request

Occurs if the request is missing required fields or if the session ID is invalid.

```json
{
  "error": "question field is required"
}
```

```json
{
  "error": "session_id field is required"
}
```

```json
{
  "error": "session not found"
}
```

#### 500 Internal Server Error

```json
{
  "error": "an unexpected error occurred"
}
```

## Example Request

```sh
POST /chat/extract_topic_and_tags
Content-Type: application/json

{
  "question": "How did Microsoft and Apple do in the last earnings season?",
  "session_id": "abc123xyz"
}
```

This request would result in a response identifying the topic as "stock\_performance" and extracting the relevant stock symbols and financial context.

---

# User Context API

## Endpoints

### POST `/user_context`

Creates a new user context, including user profile information and portfolio holdings. This can be useful to personalize the responses that the chatbot will give by passing the `user_id` in the tags of `POST /chat` endpoint.

## Request Body

| Field            | Type              | Required | Description                                                    |
| ---------------- | ----------------- | -------- | -------------------------------------------------------------- |
| `user_id`        | string            | Yes      | Unique identifier for the user.                                |
| `user_profile`   | object            | Yes      | Arbitrary key-value pairs containing user profile information. |
| `user_portfolio` | array of holdings | Yes      | List of portfolio holdings for the user.                       |

### User Portfolio Holding (Item in `user_portfolio`)

| Field                  | Type   | Required | Description                                                  |
| ---------------------- | ------ | -------- | ------------------------------------------------------------ |
| `asset_class`          | string | Yes      | Asset class, must be one of: `"stock"`, `"etf"`, `"crypto"`. |
| `symbol`               | string | No       | Ticker symbol of the asset.                                  |
| `name`                 | string | No       | Name of the asset.                                           |
| `quantity`             | number | Yes      | Number of units held.                                        |
| `portfolio_percentage` | number | Yes      | Percentage of the asset in the total portfolio.              |

> At least one of `symbol` or `name` is required for each holding.

> `user_profile` is a dynamic key value field that you can pass any information that the llm could find useful to give a more personalized response to the user. Whatever is passed in the user_profile will be given as is in the prompt that will be used to generate the response for the user.

### Example Request Body

```json
{
  "user_id": "user_123",
  "user_profile": {
    "risk_tolerance": "moderate",
    "age": 35
  },
  "user_portfolio": [
    {
      "asset_class": "stock",
      "symbol": "AAPL",
      "quantity": 10,
      "portfolio_percentage": 50
    },
    {
      "asset_class": "crypto",
      "name": "Bitcoin",
      "quantity": 0.5,
      "portfolio_percentage": 50
    }
  ]
}
```

## Response

### Success Response (201 Created)

Returns the created user context.

```json
{
  "user_id": "user_123",
  "user_profile": {
    "risk_tolerance": "moderate",
    "age": 35
  },
  "user_portfolio": [
    {
      "asset_class": "stock",
      "symbol": "AAPL",
      "name": "",
      "quantity": 10,
      "portfolio_percentage": 50
    },
    {
      "asset_class": "crypto",
      "symbol": "",
      "name": "Bitcoin",
      "quantity": 0.5,
      "portfolio_percentage": 50
    }
  ]
}
```

### Error Responses

#### 400 Bad Request

```json
{
  "error": "user_id is required"
}
```

```json
{
  "error": "user context for user_123 already exists"
}
```

#### 500 Internal Server Error

```json
{
  "error": "internal server error"
}
```

---

### PUT `/user_context`

Updates an existing user context.

## Request Body

Same as `POST /user_context`.

### Example Request Body

```json
{
  "user_id": "user_123",
  "user_profile": {
    "risk_tolerance": "aggressive"
  },
  "user_portfolio": [
    {
      "asset_class": "stock",
      "symbol": "TSLA",
      "quantity": 5,
      "portfolio_percentage": 60
    },
    {
      "asset_class": "etf",
      "name": "S&P 500 ETF",
      "quantity": 3,
      "portfolio_percentage": 40
    }
  ]
}
```

## Response

### Success Response (200 OK)

Returns the updated user context.

```json
{
  "user_id": "user_123",
  "user_profile": {
    "risk_tolerance": "aggressive"
  },
  "user_portfolio": [
    {
      "asset_class": "stock",
      "symbol": "TSLA",
      "name": "",
      "quantity": 5,
      "portfolio_percentage": 60
    },
    {
      "asset_class": "etf",
      "symbol": "",
      "name": "S&P 500 ETF",
      "quantity": 3,
      "portfolio_percentage": 40
    }
  ]
}
```

### Error Responses

#### 400 Bad Request

```json
{
  "error": "user_id is required"
}
```

```json
{
  "error": "user context for user_123 not found"
}
```

#### 500 Internal Server Error

```json
{
  "error": "internal server error"
}
```

---

### GET `/user_context/:user_id`

Retrieves an existing user context by user ID.

## Path Parameter

| Parameter | Type   | Required | Description                     |
| --------- | ------ | -------- | ------------------------------- |
| `user_id` | string | Yes      | Unique identifier for the user. |

## Response

### Success Response (200 OK)

```json
{
  "user_id": "user_123",
  "user_profile": {
    "risk_tolerance": "moderate",
    "age": 35
  },
  "user_portfolio": [
    {
      "asset_class": "stock",
      "symbol": "AAPL",
      "name": "",
      "quantity": 10,
      "portfolio_percentage": 50
    },
    {
      "asset_class": "crypto",
      "symbol": "",
      "name": "Bitcoin",
      "quantity": 0.5,
      "portfolio_percentage": 50
    }
  ]
}
```

### Error Responses

#### 400 Bad Request

```json
{
  "error": "user context for user_123 not found"
}
```

#### 500 Internal Server Error

```json
{
  "error": "internal server error"
}
```

---

## Notes

* `user_id` is required in all requests.
* Each portfolio holding requires at least one of `symbol` or `name`.
* The `asset_class` field must be one of `"stock"`, `"etf"`, or `"crypto"`.
* Returns clear error messages when user context already exists, is not found, or input validation fails.

---


# Generate Follow-Up Questions API

## Endpoint

### POST `/follow_up_questions`

Generates a list of follow-up questions based on the current session context.

## Request Body

| Field               | Type    | Required | Description                                                             |
|--------------------|---------|----------|-------------------------------------------------------------------------|
| `session_id`        | string  | Yes      | A valid session ID obtained from the `/session` endpoint.               |
| `number_of_questions` | int  | No       | The number of follow-up questions to generate. Defaults to `5` if not provided or set to `0`. |

### Example Request Body
```json
{
  "session_id": "abc123xyz",
  "number_of_questions": 3
}
```

## Response

### Success Response (200 OK)

Returns a list of AI-generated follow-up questions relevant to the conversation context.

#### Example Response Body:
```json
{
  "follow_up_questions": [
    "Would you like to see a breakdown of the revenue sources?",
    "Should I compare this company's performance with its competitors?",
    "Do you want to explore the impact of macroeconomic trends?"
  ]
}
```

### Error Responses

#### 400 Bad Request
Occurs when required fields are missing or the session ID is invalid.

```json
{
  "error": "session_id field is required"
}
```

```json
{
  "error": "session not found"
}
```

#### 500 Internal Server Error
Returned when an unexpected server error occurs.

```json
{
  "error": "an unexpected error occurred while generating follow-up questions"
}
```

## Notes
- If `number_of_questions` is not provided or set to `0`, the service will default to returning **5** questions.
- A valid `session_id` is required and must correspond to an active chat session.
- This endpoint is useful for guiding users toward deeper exploration or next steps in their inquiry.

## Example Request
```sh
POST /follow_up_questions
Content-Type: application/json

{
  "session_id": "abc123xyz",
  "number_of_questions": 5
}
```

This request would return 5 follow-up questions tailored to the given session.

---

# Get FAQs by Topic API

## Endpoint

### GET `/faq`

Retrieves a list of frequently asked questions (FAQs) for a specific topic.

## Request Parameters

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `faq_topic`   | string | Yes      | The FAQ topic identifier. Must be one of the supported topics: `education`, `sectors`, `stock_overview`, `balance_sheet`, `income_statement`, `cash_flow`, `etfs`. |

## Response

### Success Response (200 OK)

#### Example Response Body:
```json
{
  "faq": [
    "What is the stock market?",
    "How does compound interest work?",
    "What is the difference between stocks and ETFs?"
  ]
}
```

### Error Responses

#### 400 Bad Request
Returned when the `topic` parameter is missing or invalid.

```json
{
  "error": "Missing or invalid 'topic' parameter"
}
```

#### 404 Not Found
Returned when the provided topic does not exist in the system.

```json
{
  "error": "FaqTopic for 'cryptocurrency' not found"
}
```

#### 500 Internal Server Error
Returned when an internal error occurs while fetching the FAQs.

```json
{
  "error": "An error occurred while retrieving FAQs"
}
```

## Notes
- The `topic` parameter is case-sensitive and must exactly match one of the following values:
  - `education`
  - `sectors`
  - `stock_overview`
  - `balance_sheet`
  - `income_statement`
  - `cash_flow`
  - `etfs`
- The response returns up to `faqLimit` randomly selected FAQs from the topic category.
- If the topic is not found, a `FaqTopicNotFoundError` is returned.

## Example Request
```sh
GET /faq?faq_topic=education
```

This request would return a limited set of education-related FAQs.

--- 

# Get Tickers API

## Endpoint

### GET `/tickers`

Retrieves a list of stock tickers with optional filtering, pagination, and search.

## Request Parameters

| Parameter      | Type   | Required | Description |
|----------------|--------|----------|-------------|
| `limit`        | int    | No       | Limits the number of results returned. Must be a valid integer. |
| `page`         | int    | No       | The page number for paginated results. Must be a valid integer. |
| `search_string`| string | No       | A search query to filter tickers by symbol or company name. |

## Response

### Success Response (200 OK)

#### Example Response Body:
```json
{
  "tickers": [
    {
      "symbol": "AAPL",
      "company_name": "Apple Inc."
    },
    {
      "symbol": "GOOGL",
      "company_name": "Alphabet Inc."
    }
  ]
}
```

### Error Responses

#### 400 Bad Request

Returned when `limit` or `page` is provided but is not a valid integer.

```json
{
  "error": "limit query param must be a valid integer"
}
```

```json
{
  "error": "page query param must be a valid integer"
}
```

#### 500 Internal Server Error

Returned when an internal server error occurs while retrieving tickers.

```json
{
  "error": "An unexpected error occurred while retrieving tickers"
}
```

## Notes
- If `limit` is not provided, the service may return all tickers or a default number based on internal logic.
- `search_string` can match either the `symbol` or `company_name` fields of a ticker.
- The results support pagination through the `limit` and `page` parameters.
- Tickers are returned as objects containing:
  - `symbol`: The ticker symbol of the company.
  - `company_name`: The full name of the company.

## Example Request
```sh
GET /tickers?limit=10&page=2&search_string=apple
```

This request would return the second page of up to 10 tickers that match the search string "apple".

---

# Get Sector Stocks API

## Endpoint

### GET `/sectors/stocks/:sector`

Retrieves a list of stocks belonging to a specific sector.

## Request Parameters

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `sector`  | string | Yes      | The sector identifier used to filter stocks. This should be the `url_name` field from the `/sectors` endpoint response. |

## Response

### Success Response (200 OK)

#### Example Response Body:
```json
{
  "SectorStocks": [
    {
      "Symbol": "AAPL",
      "CompanyName": "Apple Inc.",
      "MarketCap": 2500000000000
    },
    {
      "Symbol": "MSFT",
      "CompanyName": "Microsoft Corporation",
      "MarketCap": 2200000000000
    }
  ]
}
```

### Error Response (500 Internal Server Error)

#### Example Response Body:
```json
{
  "error": "An error occurred while fetching sector stocks"
}
```

## Notes
- The `sector` parameter should be a valid `url_name` from the `/sectors` endpoint response (e.g., `technology`, `finance`).
- The response returns an array of stock objects, each containing:
  - `Symbol`: The stock ticker symbol.
  - `CompanyName`: The name of the company.
  - `MarketCap`: The company's market capitalization.
- If an error occurs while fetching the stocks, an appropriate error message will be returned.

## Example Request
```sh
GET /sectors/stocks/technology
```

This request would return a list of technology sector stocks.

---

# Get Sectors API

## Endpoint

### GET `/sectors`

Retrieves a list of all available sectors and their details.

## Response

### Success Response (200 OK)

#### Example Response Body:
```json
{
  "Sectors": [
    {
      "name": "Technology",
      "url_name": "technology",
      "number_of_stocks": 150,
      "market_cap": 15000000000000,
      "dividend_yield_pct": 1.5,
      "pe_ratio": 25.4,
      "profit_margin_pct": 12.3,
      "one_year_change_pct": 15.2
    },
    {
      "name": "Finance",
      "url_name": "finance",
      "number_of_stocks": 120,
      "market_cap": 12000000000000,
      "dividend_yield_pct": 2.1,
      "pe_ratio": 18.7,
      "profit_margin_pct": 10.5,
      "one_year_change_pct": 8.4
    }
  ]
}
```

### Error Response (500 Internal Server Error)

#### Example Response Body:
```json
{
  "error": "An error occurred while fetching sectors"
}
```

## Notes
- The response returns an array of sector objects, each containing:
  - `name`: The name of the sector.
  - `url_name`: A URL-friendly version of the sector name.
  - `number_of_stocks`: The total number of stocks in the sector.
  - `market_cap`: The total market capitalization of the sector.
  - `dividend_yield_pct`: The average dividend yield percentage.
  - `pe_ratio`: The average price-to-earnings ratio.
  - `profit_margin_pct`: The average profit margin percentage.
  - `one_year_change_pct`: The percentage change in sector value over the past year.

## Example Request
```sh
GET /sectors
```

This request would return a list of all available sectors and their details.

---

# Get ETFs API

## Endpoint

### GET `/etfs`

Retrieves a list of exchange-traded funds (ETFs), optionally filtered by a search string.

## Request Parameters

| Parameter      | Type   | Required | Description |
|----------------|--------|----------|-------------|
| `search_string`| string | No       | A search query to filter ETFs by symbol or name. |

## Response

### Success Response (200 OK)

#### Example Response Body:
```json
{
  "etfs": [
    {
      "symbol": "SPY",
      "name": "SPDR S&P 500 ETF Trust",
      "asset_class": "Equity",
      "aum": 411000000000
    },
    {
      "symbol": "QQQ",
      "name": "Invesco QQQ Trust",
      "asset_class": "Equity",
      "aum": 200000000000
    }
  ]
}
```

### Error Response (500 Internal Server Error)

Returned when an internal server error occurs while retrieving ETFs.

```json
{
  "error": "An unexpected error occurred while retrieving ETFs"
}
```

## Notes
- The `search_string` parameter allows filtering by ETF `symbol` or `name`. It is case-insensitive and supports partial matches.
- The response returns a list of ETFs, each including:
  - `symbol`: The ticker symbol of the ETF.
  - `name`: The full name of the ETF.
  - `asset_class`: The asset class category (e.g., Equity, Bond, Commodity).
  - `aum`: Assets under management, represented as a float.

## Example Request
```sh
GET /etfs?search_string=nasdaq
```

This request would return ETFs whose symbol or name includes "nasdaq".

---

# Get FAQ Topics API

## Endpoint

### GET `/topics`

Retrieves a list of all available FAQ topics supported by the system.

## Request Parameters

This endpoint does **not** require any request parameters.

## Response

### Success Response (200 OK)

#### Example Response Body:
```json
{
  "topics": [
    "education",
    "sectors",
    "stock_overview",
    "stock_financials",
    "etfs",
    "news"
  ]
}
```

### Error Response (500 Internal Server Error)

Returned if an unexpected server error occurs.

```json
{
  "error": "An unexpected error occurred while retrieving topics"
}
```

## Notes
- This endpoint returns a flat list of string values representing the available FAQ topics.
- The topics returned here can be used as valid `topic` values for the `/faq` endpoint.
- These values are case-sensitive and should be passed exactly as returned when used in requests.

## Example Request
```sh
GET /topics
```

This request would return a list of all valid FAQ topic identifiers.

---


## 📞 Contact
Reach out via email: [stefanouorestis@gmail.com](mailto:stefanouorestis@gmail.com)

