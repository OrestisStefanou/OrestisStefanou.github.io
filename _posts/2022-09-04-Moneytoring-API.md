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

## Source code: https://github.com/OrestisStefanou/moneytoring

### Create an account
```
curl -X 'POST' \
  'http://127.0.0.1:8000/signup' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "username": "ironman",
  "email": "ironman@gmail.com",
  "password": "jarvissendhelp"
}'
```

### Get access token to provide in requests
```
curl -X 'POST' \
  'http://127.0.0.1:8000/token' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'grant_type=&username=ironman%40gmail.com&password=jarvissendhelp&scope=&client_id=&client_secret='
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

### How to link a bank account
1. Example to get available banks in Cyprus
```
curl -X 'GET' \
  'http://127.0.0.1:8000/institutions?country_code=CY' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access_token >'
```

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

2. Lets say we want to create a connection with bank of Cyprus
```
curl -X 'POST' \
  'http://127.0.0.1:8000/bank_connections' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access_token >' \
  -H 'Content-Type: application/json' \
  -d '{
  "institution_id": "BANKOFCYPRUS_BCYPCY2NXXX",
  "redirect_uri": "random_url.com"
}'
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
User must open the "link" field of the response to start(and finish) the bank linking procedure

3. Assuming that the user linked their account successfully we can then fetch the bank account ids

```
curl -X 'GET' \
  'http://127.0.0.1:8000/bank_connections' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access_token >'
```

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

### Fetch transactions

Lets say we want to fetch the transactions of bank account with id "7e944232-bda9-40bc-b784-660c7ab5fe78"

```
curl -X 'GET' \
  'http://127.0.0.1:8000/account_transactions/7e944232-bda9-40bc-b784-660c7ab5fe78' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access_token >'
```

There are some optional query parameters here if we wish some filtering

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

To fetch the transactions of all our bank accounts for all institutions that we have linked:

curl -X 'GET' \
  'http://127.0.0.1:8000/account_transactions' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access_token >'

Same optional query parameters exist here as well

### Categorize transactions
Let's say we want to set category=food for transaction with id 2022090401927908-1

```
curl -X 'PUT' \
  'http://127.0.0.1:8000/account_transactions/2022090401927908-1/category?category=food&set_all=false' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access_token >'
```

By setting query parameter **set_all = True** all transactions with the same information will be categorised as food 

To set custom category=Alderaan Coffe for transaction with id 2022090401927908-1

```
curl -X 'PUT' \
  'http://127.0.0.1:8000/account_transactions/2022090401927908-1/custom_category?category=Alderaan%20Coffe&set_all=false' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access_token >'
```

By setting query parameter **set_all = True** all transactions with the same information will be categorised as food 


### Aggregation
1. To get total amount that we spent for a specific account

```
curl -X 'GET' \
  'http://127.0.0.1:8000/account_transactions/7e944232-bda9-40bc-b784-660c7ab5fe78/total_spent' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access token >'
```

Response
```
{
  "total_spent": -4050
}
```

There are some optional query parameters here if we wish some filtering

- from_date: Return transactions that were booked after this date
- to_date: Return transactions that were booked before this date
- category: Return transactions that are marked with this category
- custom_category: Return transactions that are marked with this category

2. To get total amount spent for all accounts

```
curl -X 'GET' \
  'http://127.0.0.1:8000/transactions/total_spent' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access token >'
```

Response
```
{
  "total_spent": -8055
}
```

There are some optional query parameters here if we wish some filtering

- from_date: Return transactions that were booked after this date
- to_date: Return transactions that were booked before this date
- category: Return transactions that are marked with this category
- custom_category: Return transactions that are marked with this category

3. To get total amount credited for a specific account
```
curl -X 'GET' \
  'http://127.0.0.1:8000/account_transactions/7e944232-bda9-40bc-b784-660c7ab5fe78/total_credited' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access token >'
```

Response
```
{
  "total_credited": 12150
}
```

There are some optional query parameters here if we wish some filtering

- from_date: Return transactions that were booked after this date
- to_date: Return transactions that were booked before this date
- category: Return transactions that are marked with this category
- custom_category: Return transactions that are marked with this category

4. To get total amount spent for all accounts

```
curl -X 'GET' \
  'http://127.0.0.1:8000/transactions/total_credited' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer < access token >'
```

Response
```
{
  "total_credited": 24165
}
```

There are some optional query parameters here if we wish some filtering

- from_date: Return transactions that were booked after this date
- to_date: Return transactions that were booked before this date
- category: Return transactions that are marked with this category
- custom_category: Return transactions that are marked with this category
