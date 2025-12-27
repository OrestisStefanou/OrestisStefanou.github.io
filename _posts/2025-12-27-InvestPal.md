---
layout: post
read_time: true
show_date: true
title: "InvestPal: AI-Powered Investment Advisory Platform"
date: 2025-12-27
img: posts/20251227/ai_investment_advisor.jpg
tags: [investing, AI]
author: Stefanou Orestis
description: "InvestPal is a comprehensive AI-powered investment advisory ecosystem that delivers personalized financial insights and market intelligence through multiple channels."
---

# InvestPal: AI-Powered Investment Advisory Platform

## Executive Summary

InvestPal is a comprehensive AI-powered investment advisory ecosystem that delivers personalized financial insights and market intelligence through multiple channels. The platform combines cutting-edge artificial intelligence with real-time market data to provide users with professional-grade investment advice accessible through both web services and messaging platforms.

The InvestPal ecosystem provides three different layers of products:

1. **MCP Server** - Exposes investing data tools that can be used in various agent user interfaces like Claude Desktop
2. **Investor Assistant/Advisor API** - A service that provides AI-powered investment advisory capabilities
3. **Telegram Bot** - A conversational interface (@ai_investor_advisor_bot) for on-the-go investment queries

---

## Product Overview

### Value Proposition

InvestPal democratizes access to sophisticated investment analysis by combining:

- **Real-time Market Intelligence**: Access to stocks, ETFs, cryptocurrencies, economic indicators, and institutional investor portfolios
- **Personalized AI Advisory**: Context-aware recommendations tailored to individual user preferences, risk tolerance, and investment goals
- **Multi-Channel Access**: Engage with your investment advisor through web APIs or conversational Telegram interface
- **Institutional-Grade Data**: Leveraging trusted sources like Alpha Vantage and CoinGecko for reliable market information

### Target Users

- **Individual Investors**: Seeking professional-grade analysis without the cost of traditional financial advisors
- **Financial Enthusiasts**: Looking to track markets, understand trends, and make informed decisions
- **Portfolio Managers**: Requiring quick access to comprehensive market data and analysis
- **Fintech Developers**: Building investment-related applications needing a robust data and AI backend

### Key Features

#### Comprehensive Market Coverage
- **Equities**: Real-time stock data, company overviews, financial statements, and sector analysis
- **ETFs**: Detailed ETF information including holdings and performance metrics
- **Cryptocurrencies**: Live crypto data, news, and market trends from CoinGecko
- **Economic Indicators**: Historical data on GDP, inflation, unemployment, and more
- **Commodities**: Time-series data for oil, gas, and other commodities
- **Super Investor Tracking**: Monitor portfolios of institutional investors

#### Intelligent Advisory Capabilities
- **Context-Aware Conversations**: Remembers user preferences, risk tolerance, and investment history
- **Multi-LLM Support**: Choose from OpenAI GPT-4, Google Gemini, or Anthropic Claude
- **Session Management**: Persistent chat history for continuous, coherent conversations
- **Personalized Insights**: AI analyzes your specific situation and goals to provide tailored advice

#### Flexible Integration
- **RESTful API**: Full-featured web service for programmatic access
- **Telegram Bot**: Natural language interface for on-the-go investment queries
- **MCP Protocol**: Extensible tool system for integrating additional data sources

---

## Technical Architecture

### System Design Philosophy

InvestPal follows a microservices architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│                    User Layer                        │
│  (Telegram Bot, Web Client, API Consumers)          │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│              InvestPal Core Service                  │
│  (FastAPI + LangChain + AI Orchestration)           │
│  - Session Management                                │
│  - User Context Storage                              │
│  - LLM Integration (OpenAI/Google/Anthropic)        │
└────────────────┬────────────────────────────────────┘
                 │ MCP Protocol
┌────────────────▼────────────────────────────────────┐
│           MarketDataMcpServer                        │
│  (Go-based MCP Server)                              │
│  - Alpha Vantage Integration                         │
│  - CoinGecko Integration                             │
│  - Data Caching & Rate Limiting                      │
│  - Economic Indicators                               │
└─────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. MarketDataMcpServer (Data Layer)

**Technology Stack:**
- Language: Go 1.21+
- Protocol: Model Context Protocol (MCP)
- Data Providers: Alpha Vantage, CoinGecko
- Storage: BadgerDB or MongoDB
- Caching: Configurable TTL-based caching

**Purpose:**
Provides standardized access to diverse financial data sources through the Model Context Protocol. Handles API rate limiting, data caching, and abstracts away the complexity of individual data provider APIs.

**Key Capabilities:**
- 20+ financial data tools covering stocks, ETFs, cryptocurrencies, economic indicators, and more
- Intelligent caching to optimize performance and API quota usage
- Support for both embedded and distributed storage options
- User context management for personalized data retrieval

#### 2. InvestPal Core (Intelligence Layer)

**Technology Stack:**
- Framework: FastAPI (Python)
- AI Framework: LangChain
- Database: MongoDB
- Supported LLMs: OpenAI GPT-4, Google Gemini, Anthropic Claude

**Purpose:**
Orchestrates AI-powered investment conversations by combining market data with user context. Acts as the intelligence layer that transforms raw financial data into personalized insights.

**Key Capabilities:**
- RESTful API for user context, session, and chat management
- AI agent that autonomously selects and uses appropriate data tools
- Persistent conversation history with configurable context windows
- Multi-LLM support for flexibility in AI model selection

#### 3. InvestPalTelegramBot (User Interface Layer)

**Technology Stack:**
- Framework: python-telegram-bot (async)
- Language: Python 3.14+
- Database: SQLite

**Purpose:**
Provides a conversational interface through Telegram, making AI-powered investment advice accessible via messaging. Handles user interactions and formats responses for optimal readability.

**Key Capabilities:**
- Webhook-based integration with Telegram
- Automatic formatting of AI responses for Telegram
- Session mapping and continuity across conversations
- Message splitting for long responses

---

## Data Flow & Integration

### Typical User Interaction Flow

1. **User Initiates Query**
   - Via Telegram message or API call
   - Example: "Should I invest in tech stocks right now?"

2. **Session Context Retrieval**
   - InvestPal Core retrieves user's investment profile
   - Loads conversation history for context

3. **AI Analysis & Tool Selection**
   - LangChain agent analyzes query
   - Determines needed data (e.g., sector performance, market news)
   - Generates tool calls to MarketDataMcpServer

4. **Data Gathering**
   - MCP server fetches real-time market data
   - Applies caching when appropriate
   - Returns structured financial information

5. **Intelligent Synthesis**
   - AI combines market data with user context
   - Considers risk tolerance, investment goals, and preferences
   - Generates personalized, actionable advice

6. **Response Delivery**
   - Formatted response returned to user
   - Conversation state persisted for continuity

### Model Context Protocol (MCP)

MCP serves as the bridge between the AI agent and financial data sources, providing a standardized way for the AI to discover and use available tools. This allows InvestPal Core to dynamically access market data without hardcoding specific data provider logic, making the system extensible and maintainable.

---

## Configuration & Deployment

Detailed configuration options and deployment instructions for each service are available in their respective README files in the GitHub repositories.

---

## Development

Development setup instructions and workflows for each service are available in their respective README files in the GitHub repositories.

---

## Contributing & Support

### Repository Links
- MarketDataMcpServer: https://github.com/OrestisStefanou/MarketDataMcpServer
- InvestPal Core: https://github.com/OrestisStefanou/InvestPal
- Telegram Bot: https://github.com/OrestisStefanou/InvestPalTelegramBot

### Documentation
Each repository contains detailed setup instructions and API documentation. The InvestPal Core service provides interactive API documentation via FastAPI's Swagger UI at the `/docs` endpoint.
