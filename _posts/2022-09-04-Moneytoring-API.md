---
layout: post
read_time: true
show_date: true
title: "Moneytoring API service"
date: 2022-09-04
img: posts/20220904/Open-banking.jpeg
tags: [open-banking, api, python]
author: Stefanou Orestis
description: "An API to fetch, categorise and aggregate bank account transactions"
---
Moneytoring is an api that provides functionality for users to link their bank accounts, see their transactions, categorize them with standard or custom categories and aggregate them

### Create an account
Send a POST request to **/signup** endpoint with body(json)
```
{
    "username": string,
    "email": string,
    "password": string,
}
```

### Get access token to provide in requests
Send a POST request to **/token** endpoint with body(form-data)
Set header 'Content-Type: application/x-www-form-urlencoded'
```
{
    "username": string,
    "password": string,
}
```
In the response you will receive the access_token
```
{
  "access_token": "<access_token>",
  "token_type": "bearer"
}
```

For the rest of the endpoints an access token should be provided in the headers
-H 'Authorization: Bearer < access_token >'

## How to link a bank account
1. Send a GET request to **/institutions** endpoint with query parameter the country code of the bank you want to link.
Example to get available banks in Cyprus
GET /institutions?country_code=CY

Response
```
[
  {
    "id": "AIRWALLEX_AIPTAU32",
    "name": "Airwallex",
    "bic": "AIPTAU32",
    "transaction_total_days": 730,
    "logo": "https://cdn.nordigen.com/ais/AIRWALLEX_AIPTAU32_1.png"
  },
  {
    "id": "BANKOFCYPRUS_BCYPCY2NXXX",
    "name": "Bank of Cyprus",
    "bic": "BCYPCY2NXXX",
    "transaction_total_days": 730,
    "logo": "https://cdn.nordigen.com/ais/BANKOFCYPRUS_BCYPCY2NXXX.png"
  }
```

Lets say we want to link our bank account of bank of Cyprus

2. Send a POST request to **/bank_connections** endpoint with body(json)
```
{
  "institution_id": "BANKOFCYPRUS_BCYPCY2NXXX",
  "redirect_uri": "some_url"    -> Where to redirect the user after they finish with the linking procedure
}
```

Response
```
{
  "id": "d2dee8cf-e9c3-4e72-afd6-ae5f801a3ab5",
  "institution_name": "Bank of Cyprus",
  "link": "connection_link",
  "status": "pending",
  "accepted_at": "string",
  "expires_at": "string",
  "max_historical_days": 0,
  "bank_accounts": []
}
```
User must open the "link" field if the response to start(and finish) the bank linking procedure

Assuming that the user linked their account successfully we can then fetch the bank account ids

3. Send a GET request to **/bank_connections** endpoints
Response
```
[
  {
    "id": "d2dee8cf-e9c3-4e72-afd6-ae5f801a3ab5",
    "institution_name": "Bank of Cyprus",
    "link": "",
    "status": "created",
    "accepted_at": "2022-07-25T19:15:44.301672Z",
    "expires_at": "2022-10-23",
    "max_historical_days": 90,
    "bank_accounts": [
      {
        "account_id": "7e944232-bda9-40bc-b784-660c7ab5fe78",
        "name": "Main Account",
        "currency": "EUR"
      },
      {
        "account_id": "99a0bfe2-0bef-46df-bff2-e9ae0c6c5838",
        "name": "Main Account",
        "currency": "EUR"
      }
    ]
  }
]
```

## Fetch transactions

Lets say we want to fetch the transactions of bank account with id "7e944232-bda9-40bc-b784-660c7ab5fe78"

We send a GET request to **/account_transactions/7e944232-bda9-40bc-b784-660c7ab5fe78** endpoint. There are some optional query parameters here if we wish some filtering

- from_date: Return transactions that were booked after this date
- to_date: Return transactions that were booked before this date
- category: Return transactions that are marked with this category
- custom_category: Return transactions that are marked with this category

Response
```
[
  {
    "id": "2022090401927907-1",
    "account_id": "7e944232-bda9-40bc-b784-660c7ab5fe78",
    "amount": "45.00",
    "currency": "EUR",
    "information": "For the support of Restoration of the Republic foundation",
    "code": "PMNT",
    "created_date": "2022-09-04",
    "booking_date": "2022-09-04",
    "debtor_name": "MON MOTHMA",
    "category": null,
    "custom_category": null
  },
  {
    "id": "2022090401927908-1",
    "account_id": "7e944232-bda9-40bc-b784-660c7ab5fe78",
    "amount": "-15.00",
    "currency": "EUR",
    "information": "PAYMENT Alderaan Coffe",
    "code": "PMNT",
    "created_date": "2022-09-04",
    "booking_date": "2022-09-04",
    "debtor_name": null,
    "category": null,
    "custom_category": null
  },
]
```

If we want to fetch the transactions of all our bank accounts for all institutions that we have linked:
We send a GET request to **/account_transactions** endpoint. Same optional query parameters exist here as well

## Categorize transactions
