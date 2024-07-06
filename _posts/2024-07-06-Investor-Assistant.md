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
- [Prompt engineering](#prompt-engineering)
    - [Database schema](#database-schema)
    - [Dynamic few-shot examples](#dynamic-few-shot-examples)
    - [System prefix](#system-prefix)
    - [Full prompt](#full-prompt)
- [Investor Agent class](#investor-agent-class)
    - [Conversation History](#conversation-history)


## Project Goal
The goal of the project is to build a question/answering system over an SQL database that contains various financial data. We will also expose this chatbot through a web api so that multiple users can use it.

## Solution
The solution will be broken into these steps
- Prompt engineering
- Creating an sql agent using Langchain
- Creating chat message history
- Exposing the agent through a web api

## Prompt Engineering
#### Database schema
The database that our agent will query is an sqlite3 database that has the following schema:
```sql
CREATE TABLE world_indices_time_series (
    index_name TEXT NOT NULL,
    open_price REAL NOT NULL,
    high_price REAL NOT NULL,
    low_price REAL NOT NULL,
    close_price REAL NOT NULL,
    volume REAL NOT NULL,
    registered_date TEXT NOT NULL,
    registered_date_ts INT NOT NULL
);

CREATE TABLE economic_indicator_time_series (
    indicator_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    registered_date TEXT NOT NULL,
    registered_date_ts INT NOT NULL
);

CREATE TABLE stock_overview (
    symbol TEXT NOT NULL,
    sector TEXT NOT NULL,
    industry TEXT,
    market_cap FLOAT,
    ebitda FLOAT,
    pe_ratio FLOAT,
    forward_pe_ratio FLOAT,
    trailing_pe_ratio FLOAT,
    peg_ratio FLOAT,
    book_value FLOAT,
    divided_per_share FLOAT,
    dividend_yield FLOAT,
    eps FLOAT,
    diluted_eps FLOAT,
    revenue_per_share FLOAT,
    profit_margin FLOAT,
    operating_margin FLOAT,
    return_on_assets FLOAT,
    return_on_equity FLOAT,
    revenue FLOAT,
    gross_profit FLOAT,
    quarterly_earnings_growth_yoy FLOAT,
    quarterly_revenue_growth_yoy FLOAT,
    target_price FLOAT,
    beta FLOAT,
    price_to_sales_ratio FLOAT,
    price_to_book_ratio FLOAT,
    ev_to_revenue FLOAT,
    ev_to_ebitda FLOAT,
    outstanding_shares FLOAT,
    registered_date TEXT NOT NULL,
    registered_date_ts INT NOT NULL
);

CREATE TABLE income_statement (
    symbol TEXT NOT NULL,
    fiscal_date_ending TEXT NOT NULL,
    reported_currency TEXT,
    gross_profit FLOAT,
    total_revenue FLOAT,
    cost_of_revenue FLOAT,
    cost_of_goods_and_services_sold FLOAT,
    operating_income FLOAT,
    selling_general_and_administrative FLOAT,
    research_and_development FLOAT,
    operating_expenses FLOAT,
    investment_income_net FLOAT,
    net_interest_income FLOAT,
    interest_income FLOAT,
    interest_expense FLOAT,
    non_interest_income FLOAT,
    other_non_operating_income FLOAT,
    depreciation FLOAT,
    depreciation_and_amortization FLOAT,
    income_before_tax FLOAT,
    income_tax_expense FLOAT,
    interest_and_debt_expense FLOAT,
    net_income_from_continuing_operations FLOAT,
    comprehensive_income_net_of_tax FLOAT,
    ebit FLOAT,
    ebitda FLOAT,
    net_income FLOAT
);

CREATE TABLE balance_sheet (
    symbol TEXT NOT NULL,
    fiscal_date_ending TEXT NOT NULL,
    reported_currency TEXT,
    total_assets FLOAT,
    total_current_assets FLOAT,
    cash_and_cash_equivalents_at_carrying_value FLOAT,
    cash_and_short_term_investments FLOAT,
    inventory FLOAT,
    current_net_receivables FLOAT,
    total_non_current_assets FLOAT,
    property_plant_equipment FLOAT,
    accumulated_depreciation_amortization_ppe FLOAT,
    intangible_assets FLOAT,
    intangible_assets_excluding_goodwill FLOAT,
    goodwill FLOAT,
    investments FLOAT,
    long_term_investments FLOAT,
    short_term_investments FLOAT,
    other_current_assets FLOAT,
    other_non_current_assets FLOAT,
    total_liabilities FLOAT,
    total_current_liabilities FLOAT,
    current_accounts_payable FLOAT,
    deferred_revenue FLOAT,
    current_debt FLOAT,
    short_term_debt FLOAT,
    total_non_current_liabilities FLOAT,
    capital_lease_obligations FLOAT,
    long_term_debt FLOAT,
    current_long_term_debt FLOAT,
    long_term_debt_noncurrent FLOAT,
    short_long_term_debt_total FLOAT,
    other_current_liabilities FLOAT,
    other_non_current_liabilities FLOAT,
    total_shareholder_equity FLOAT,
    treasury_stock FLOAT,
    retained_earnings FLOAT,
    common_stock FLOAT,
    common_stock_shares_outstanding FLOAT
);

CREATE TABLE cash_flow(
    symbol TEXT NOT NULL,
    fiscal_date_ending TEXT NOT NULL,
    reported_currency TEXT,
    operating_cashflow FLOAT,
    payments_for_operating_activities FLOAT,
    proceeds_from_operating_activities FLOAT,
    change_in_operating_liabilities FLOAT,
    change_in_operating_assets FLOAT,
    depreciation_depletion_and_amortization FLOAT,
    capital_expenditures FLOAT,
    change_in_receivables FLOAT,
    change_in_inventory FLOAT,
    profit_loss FLOAT,
    cashflow_from_investment FLOAT,
    cashflow_from_financing FLOAT,
    proceeds_from_repayments_of_short_term_debt FLOAT,
    payments_for_repurchase_of_common_stock FLOAT,
    payments_for_repurchase_of_equity FLOAT,
    payments_for_repurchase_of_preferred_stock FLOAT,
    dividend_payout FLOAT,
    dividend_payout_common_stock FLOAT,
    dividend_payout_preferred_stock FLOAT,
    proceeds_from_issuance_of_common_stock FLOAT,
    proceeds_from_issuance_of_long_term_debt_and_capital_securities_net FLOAT,
    proceeds_from_issuance_of_preferred_stock FLOAT,
    proceeds_from_repurchase_of_equity FLOAT,
    proceeds_from_sale_of_treasury_stock FLOAT,
    change_in_cash_and_cash_equivalents FLOAT,
    change_in_exchange_rate FLOAT,
    net_income FLOAT
);

CREATE TABLE stock_time_series (
    symbol TEXT NOT NULL,
    open_price REAL NOT NULL,
    high_price REAL NOT NULL,
    low_price REAL NOT NULL,
    close_price REAL NOT NULL,
    volume REAL NOT NULL,
    dividend_amount REAL NOT NULL,
    registered_date TEXT NOT NULL,
    registered_date_ts INT NOT NULL
);

CREATE TABLE super_investor_portfolio_holding (
    super_investor TEXT NOT NULL,
    stock TEXT NOT NULL,
    pct_of_portfolio REAL NOT NULL,
    shares REAL NOT NULL,
    reported_price TEXT NOT NULL,
    value TEXT NOT NULL
);

CREATE TABLE super_investor_portfolio_sector_analysis (
    super_investor TEXT NOT NULL,
    sector_name TEXT NOT NULL,
    sector_pct REAL NOT NULL
);

CREATE TABLE super_investor_grand_portfolio (
    stock TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ownership_count INT NOT NULL
);
```

#### Dynamic few-shot examples
Including examples of natural language questions being converted to valid sql queries against our database in the prompt will often improve model performance, especially for complex queries. Also in cases where tables may have some overlapping information, for example in our case `stock_overview` table contains a `revenue` column and in the `income_statement` table we have a `total_revenue` column. Depending on the question we may want our agent sometimes to get the revenue from `stock_overview` and some other times from `income_statement`.
We will use the following examples:

```python
examples = [
    {
        "input": "Find all the balance sheets of symbol 'AAPL' and 'META' in year 2023.",
        "query": """SELECT * FROM balance_sheet WHERE symbol IN ('AAPL', 'META') AND strftime('%Y', fiscal_date_ending) = '2023';""",
    },
    {
        "input": "Find all the income statements of symbol 'AAPL' in year 2023.",
        "query": """SELECT * FROM income_statement WHERE symbol = 'AAPL' AND strftime('%Y', fiscal_date_ending) = '2023';""",
    },
    {
        "input": "What are the available sectors?",
        "query": "SELECT DISTINCT sector FROM stock_overview WHERE sector is not null;",
    },
    {
        "input": "Can you compare the revenue of 'AAPL' and 'GOOGL' in year 2023?",
        "query": """SELECT symbol, total_revenue, fiscal_date_ending FROM income_statement WHERE symbol IN ('AAPL', 'GOOGL') AND strftime('%Y', fiscal_date_ending) = '2023';""",
    },
    {
        "input": "Which company from the 'Technology' sector had the most current assets in 2023?",
        "query": """SELECT b.symbol, b.total_current_assets
                    FROM balance_sheet b
                    INNER JOIN stock_overview s
                    ON s.symbol = b.symbol
                    WHERE s.sector = 'TECHNOLOGY' AND strftime('%Y', b.fiscal_date_ending) = '2023'
                    ORDER BY b.total_current_assets DESC LIMIT 1;
                """,
    },
    {
        "input": "Which company from the Technology sector had the highest total revenue to cost of revenue ratio in 2023?",
        "query": """
                SELECT i.symbol, i.total_revenue, i.cost_of_revenue, i.total_revenue/i.cost_of_revenue AS revenue_to_cost_ratio
                FROM income_statement i
                INNER JOIN stock_overview s
                ON i.symbol = s.symbol
                WHERE strftime('%Y', fiscal_date_ending) = '2023' AND s.sector = 'TECHNOLOGY'
                ORDER BY revenue_to_cost_ratio DESC
                LIMIT 1;
            """
    },
    {
        "input": "Which symbol had the highest average total revenue in 2023 in the TECHNOLOGY sector?",
        "query": """
                SELECT i.symbol, AVG(i.total_revenue) AS avg_total_revenue
                FROM income_statement i
                INNER JOIN stock_overview o
                ON i.symbol = o.symbol
                WHERE strftime('%Y', i.fiscal_date_ending) = '2023' AND o.sector = 'TECHNOLOGY'
                GROUP BY i.symbol
                ORDER BY avg_total_revenue DESC
                LIMIT 1
            """
    },
    {
        "input": "Which symbol had the highest average total revenue in 2023 from each sector?",
        "query": """
            WITH avg_revenue_per_company AS (
                SELECT
                    i.symbol,
                    AVG(i.total_revenue) AS avg_revenue,
                    s.sector
                FROM
                    income_statement i
                INNER JOIN
                    stock_overview s ON i.symbol = s.symbol
                WHERE strftime('%Y', i.fiscal_date_ending) = '2023'
                GROUP BY
                    i.symbol, s.sector
            ),
            max_avg_revenue_per_sector AS (
                SELECT
                    sector,
                    MAX(avg_revenue) AS max_avg_revenue
                FROM
                    avg_revenue_per_company
                GROUP BY
                    sector
            )
            SELECT
                arc.symbol,
                arc.sector,
                arc.avg_revenue
            FROM
                avg_revenue_per_company arc
            JOIN
                max_avg_revenue_per_sector marp
            ON
                arc.sector = marp.sector AND arc.avg_revenue = marp.max_avg_revenue;
        """
    },
    {
        "input": "Which sector had the highest total revenue in 2023?",
        "query": """
            WITH total_revenue_per_company AS (
                SELECT
                    i.symbol,
                    SUM(i.total_revenue) AS total_revenue,
                    s.sector
                FROM
                    income_statement i
                INNER JOIN
                    stock_overview s ON i.symbol = s.symbol
                WHERE strftime('%Y', i.fiscal_date_ending) = '2023'
                GROUP BY
                    i.symbol, s.sector
            ),
            total_revenue_per_sector AS (
                SELECT
                    sector,
                    SUM(total_revenue) AS total_revenue
                FROM
                    total_revenue_per_company
                GROUP BY
                    sector
            )
            SELECT
                sector,
                total_revenue
            FROM
                total_revenue_per_sector
            ORDER BY
                total_revenue DESC
            LIMIT 1;
        """
    },
    {
        "input": "Which symbol has the most cash?",
        "query": """
            WITH latest_fiscal_date AS (
                SELECT
                    symbol,
                    MAX(fiscal_date_ending) AS latest_fiscal_date
                FROM
                    balance_sheet
                GROUP BY
                    symbol
            ),
            latest_cash_equivalents AS (
                SELECT
                    bs.symbol,
                    bs.cash_and_cash_equivalents_at_carrying_value,
                    lfd.latest_fiscal_date
                FROM
                    balance_sheet bs
                JOIN
                    latest_fiscal_date lfd
                ON
                    bs.symbol = lfd.symbol AND bs.fiscal_date_ending = lfd.latest_fiscal_date
            )
            SELECT
                symbol,
                cash_and_cash_equivalents_at_carrying_value
            FROM
                latest_cash_equivalents
            ORDER BY
                cash_and_cash_equivalents_at_carrying_value DESC
            LIMIT 1;
        """
    },
    {
        "input": "Which symbol has the most short term debt?",
        "query": """
            WITH latest_fiscal_date AS (
                SELECT
                    symbol,
                    MAX(fiscal_date_ending) AS latest_fiscal_date
                FROM
                    balance_sheet
                GROUP BY
                    symbol
            ),
            latest_short_term_debt AS (
                SELECT
                    bs.symbol,
                    bs.short_term_debt,
                    lfd.latest_fiscal_date
                FROM
                    balance_sheet bs
                JOIN
                    latest_fiscal_date lfd
                ON
                    bs.symbol = lfd.symbol AND bs.fiscal_date_ending = lfd.latest_fiscal_date
            )
            SELECT
                symbol,
                short_term_debt
            FROM
                latest_short_term_debt
            ORDER BY
                short_term_debt DESC
            LIMIT 1;
        """
    },
    {
        "input": "Which symbol had the most cash in 2023?",
        "query": """SELECT symbol, cash_and_cash_equivalents_at_carrying_value FROM balance_sheet WHERE strftime('%Y', fiscal_date_ending) = '2023' ORDER BY cash_and_cash_equivalents_at_carrying_value DESC LIMIT 1;""",
    },
    {
        "input": "Compare the price movement of AAPL and MSFT the last year",
        "query": """"
                    SELECT s1.symbol, s1.close_price AS aapl_close_price, s2.close_price AS msft_close_price, s1.registered_date
                    FROM stock_time_series s1
                    JOIN stock_time_series s2 ON s1.registered_date = s2.registered_date
                    WHERE s1.symbol = 'AAPL' AND s2.symbol = 'MSFT'
                    ORDER BY s1.registered_date_ts DESC
                    LIMIT 100;
                """,
    },
    {
        "input": "Which sector had the highest average quarterly earnings growth year over year?",
        "query": """"
                SELECT sector
                FROM stock_overview
                WHERE sector!=''
                GROUP BY sector
                ORDER BY AVG(quarterly_earnings_growth_yoy)
                LIMIT 1
            """,
    },
    {
        "input": "For which superinvestors do you have portfolio data?",
        "query": """"
                SELECT DISTINCT super_investor FROM super_investor_portfolio_holding;
            """,
    },
    {
        "input": "Can you give me the portfolio holdings of super investor Warren Buffet?",
        "query": """"
                SELECT * FROM super_investor_portfolio_holding WHERE super_investor='Warren Buffet';
            """,
    },
    {
        "input": "Can you give me the portfolio sector analysis of super investor Warren Buffet?",
        "query": """"
                SELECT * FROM super_investor_portfolio_sector_analysis WHERE super_investor='Warren Buffet';
            """,
    },
    {
        "input": "Which are the top 10 most hold stocks from super investors?",
        "query": """"
                SELECT * FROM super_investor_grand_portfolio ORDER BY ownership_count DESC LIMIT 10;
            """,
    }
]
```

If we have enough examples, we may want to only include the most relevant ones in the prompt, either because they don't fit in the model's context window or because the long tail of examples distracts the model. And specifically, given any input we want to include the examples most relevant to that input.

We can do just this using an `ExampleSelector`. In this case we'll use a `SemanticSimilarityExampleSelector`, which will store the examples in the vector database of our choosing. At runtime it will perform a similarity search between the input and our examples, and return the most semantically similar ones:

```python
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(openai_api_key=settings.openai_key),
    FAISS,
    k=5,
    input_keys=["input"],
)
```
We are using the `FAISS` vector store to store the embeddings that will be created using the `OpenAIEmbeddings` class. Since we are using `k=5` the example selector will find the 5 most similar examples with the user input.

#### Table definitions and example rows
In most SQL chains, we'll need to feed the model at least part of the database schema. Without this it won't be able to write valid queries. Here we will use `SQLDatabase.get_context`, which provides available tables and their schemas:

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{settings.db_path}")
context = db.get_context()["table_info"]
```

#### System prefix
```python
system_prefix = """You are a financial advisor agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
If you get an error while executing a query, rewrite the query and try again.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database or investing related, just return "I don't know" as the answer.
The database context is as follows:
{context}

Here are some examples of user inputs and their corresponding SQL queries:"""
```

#### Full prompt
To tie all the above together we will use the `FewShotPromptTemplate` along with the `ChatPromptTemplate`.

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k", "context"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```

## Investor Agent Class
Now that we have finished with the prompt we can create the sql agent using the `create_sql_agent` function that comes with langchain. We will create an abstraction layer on top of the sql agent that will be used later in the web api

```python
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages.base import BaseMessage

from app import settings
from analytics.chatbot.prompt import (
    full_prompt,
    context
)

class InvestorAgent:
    def __init__(
        self,
        temperature: float = 0.4,
        model: str = "gpt-3.5-turbo-0125"
    ) -> None:
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature, 
            openai_api_key=settings.openai_key
        )
        self._stock_db = SQLDatabase.from_uri(f"sqlite:///{settings.db_path}")
        self._sql_agent = create_sql_agent(
            llm=self._llm,
            toolkit=SQLDatabaseToolkit(db=self._stock_db, llm=self._llm),   # A set of tools to interact with the database
            prompt=full_prompt, # prompt template that we defined above
            verbose=True,   # Show what sql queries that agent is running to get the results
            agent_executor_kwargs={"handle_parsing_errors":True},   # Handle any parsing errors that may happen
            agent_type="tool-calling",
            max_iterations=5    # The agent will execute at max 5 queries to get the answer
        )
        self._conversation_db = SQLDatabase.from_uri(f"sqlite:///{settings.chatbot_db_path}") 
```

A few things that are happening here:
1. Create the llm that will be used for the sql agent, in our case is the `gpt-3.5-turbo-0125` of OpenAI
2. Load the `SQLDatabase` that our agent will use
3. Create the sql agent
4. Load the `SQLDatabase` that we will use to store the conversation history (More on this below)

### Conversation history
In order to maintain message history in case a user wants to continue the conversation at some point in the future we have to store it somewhere. Langchain comes with a helper class that serves this exact purpose `SQLChatMessageHistory`. Below we create the `chat` that takes as parameters the question that the user asked and the session_id which is an identifier for the session (conversation) thread that these input messages correspond to. This allows you to maintain several conversations/threads with the same chain at the same time. The method will return the response that the agent gives.

```python
def chat(self, question: str, session_id: str) -> list[BaseMessage]:
    chat_message_history = SQLChatMessageHistory(
        session_id=session_id, 
        connection=self._conversation_db._engine
    )
    if len(chat_message_history.messages) > 0:
        first_question = chat_message_history.messages[0]
    else:
        first_question = question

    chat_message_history.add_user_message(question)
    response = self._sql_agent.invoke(
        {
            "input": str(first_question),
            "top_k": 5,
            "dialect": "SQLite",
            "context": context,
            "agent_scratchpad": [],
            "messages": chat_message_history.messages,
        }
    )
    output = response["output"]
    chat_message_history.add_ai_message(output)
    return chat_message_history.messages
```