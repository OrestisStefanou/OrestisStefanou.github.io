---
layout: post
read_time: true
show_date: true
title: "Google Drive Service"
date: 2021-10-25
img: posts/20211025/Google-Drive.jpg
tags: [saas, api]
author: Stefanou Orestis
description: "An api that works on top of google drive api to provide simpler endpoints for basic operations"
---
Google Drive Service is an api that works on top of Google Drive API to provide simpler endpoints for basic operations like listing the files of a user,downloading and uploading files and handling file permissions

## Who is Google Drive Service for
If you have a service that needs access to your user's google drive or you are just trying to automate some google drive operations, then Google Drive Service is for you

## How to use
Google Drive api uses oauth2 authentication,so in order to get access to your client's google drive you need an access token to perform a request on the user’s behalf.To get an access token we send a url to the user where they have to accept that your service can have access to their google drive account.After the user accepts,google returns to the user an authentication code which the user has to send back to us in order to generate the access token.In the diagram below we can see the authentication flow( 'Requesting Service' is the service that needs to access the google drive of it's users and is using google drive service to achieve that)

<center><img src='./assets/img/posts/20211025/auth_flow.png'></center>

## Endpoints
### Base url:https://radiant-gorge-35067.herokuapp.com
### Authentication Endpoints
- GET /v1/authenticationURL  
Endpoint to get the url that the user needs to authenticate their google drive account
	- Query Parameters
		- scope:string (The scope of the permissions you need access to).Available values:
			- DriveScope(Default) -> See, edit, create, and delete all of your Google Drive files
			- DriveAppdataScope -> See, create, and delete its own configuration data in your Google Drive
			- DriveFileScope -> See, edit, create, and delete only the specific Google Drive files you use with this app
			- DriveMetadataScope -> View and manage metadata of files in your Google Drive
			- DriveMetadataReadonlyScope -> See information about your Google Drive files
			- DrivePhotosReadonlyScope -> View the photos, videos and albums in your Google Photos
			- DriveReadonlyScope -> See and download all your Google Drive files
			- DriveScriptsScope -> Modify your Google Apps Script scripts' behavior
	- Response
		- Status Code:200
		- Data(JSON)
			```
			{
				"message":string,
				"authURL":string
			}
			```
	- Example in Python
		```
		url = f"{baseURL}/authenticationURL"
		payload = {'scope': 'DriveScope'}
		r = requests.get(url,params=payload)
		```
  

- POST /v1/token  
Endpoint to get access token using the authentication code given by the user
	- Request Data
		- code:string -> authentication code given by the user
	- Headers
		- Content-type: application/x-www-form-urlencoded
	- Response
		- Status Code:200
		- Data(JSON)
			```
			{
				"message":string,
				"AccessToken":{
					"access_token":string,
					"token_type":string,
					"refresh_token":string,
					"expiry":time
				}
			}
			```			
	- Python Example
			```
			url = f"{baseURL}/token"
			payload = {'code': auth_code}
			headers = {'Content-type': 'application/x-www-form-urlencoded'}
			r = requests.post(url, data=payload,headers=headers)
			response = r.json()
			if r.status_code == 200:
				json_string = json.dumps(response['AccessToken'])
				#Save the token for future requests
				f = open("token.json", "w")
				f.write(json_string)
				f.close()
			```				