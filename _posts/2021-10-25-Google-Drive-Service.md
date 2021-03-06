---
layout: post
read_time: true
show_date: true
title: "Google Drive Service"
date: 2021-10-25
img: posts/20211025/Google-Drive.jpg
tags: [saas, api, golang]
author: Stefanou Orestis
description: "An api that works on top of google drive api to provide simpler endpoints for basic operations"
---
Google Drive Service is an api that works on top of Google Drive API to provide simpler endpoints for basic operations like listing the files of a user,downloading and uploading files and handling file permissions

## Who is Google Drive Service for
If you have a service that needs access to your user's google drive or you are just trying to automate some google drive operations, then Google Drive Service is for you

## Source code: https://github.com/OrestisStefanou/google-drive-service

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
### Files Endpoints
- GET /v1/files  
Get user's files metadata
	- Query parameters
		- query:string -> To filter the files you want in the response
			Visit [google drive query examples](https://developers.google.com/drive/api/v3/search-files) for more information on how to query
	- Headers
		- Authorization: \<access_token> (access_token in string format)
	- Response
		- Status Code:200
		- Data(JSON)
			```
			{
				"Files":[
					{
					  "id": string,
					  "mimeType": string,
					  "name": string,
					  "parents": [
					    string
					  ],
					  "size": string,
					  "webContentLink": string,
					  "webViewLink": string
					}
				]
			}
			```			
	- Python Example
		```
		url = f"{baseURL}/files"
		try:
			f = open("token.json", "r")
			access_token = f.read()
			f.close()
		except:
			print("Token not found")
			return
		payload = {'query': query_string}
		headers = {'Authorization': access_token}
		r = requests.get(url,headers=headers,params=payload)
		if r.status_code == 200:
			files = r.json()['Files']
			for file in files:
				print(json.dumps(file,indent=2))
		else:
			print(r.json())
		```			
- GET /v1/files/download/\<file_id>  
Endpoint to download a file with binary content
	- Headers 
		- Authorization: \<access_token> (access_token in string format)
	- Response
		- Status Code:200
		- Data (application/binary)
	 		- The bytes of the file
	- Python Example
		```
		url = f"{baseURL}/files/download/{file_id}"
		try:
			f = open("token.json", "r")
			access_token = f.read()
			f.close()
		except:
			print("Token not found")
			return
		headers = {'Authorization': access_token}
		r = requests.get(url,headers=headers)	
		if r.status_code == 200:
			#print(r.content)
			f = open(filepath,"wb")
			f.write(r.content)
			f.close()
		else:
			print(r.json())
		```			
- GET /v1/files/download_exported/\<file_id>  
Endpoint to export a file in the desired format and download it
	- Query parameters
		- mimeType:string -> mimeType that you want to export the file 
			Visit [Google Workspace formats and supported export MIME types](https://developers.google.com/drive/api/v3/ref-export-formats) for the available mimeTypes
	- Headers
		- Authorization: \<access_token> (access_token in string format)
	- Response
		- Status Code:200
		- Data (application/binary)
			- The bytes of the file
	- Python Example
		```
		url = f"{baseURL}/files/download_exported/{file_id}"
		payload = {"mimeType":mimeType}
		try:
			f = open("token.json", "r")
			access_token = f.read()
			f.close()
		except:
			print("Token not found")
			return
		headers = {'Authorization': access_token}
		r = requests.get(url,headers=headers,params=payload)	
		if r.status_code == 200:
			#print(r.content)
			f = open(filepath,"wb")
			f.write(r.content)
			f.close()
		else:
			print(r.json())
		```			

- POST /v1/files/folder  
Endpoint to create a folder in the user's drive 
	- Request data(JSON)
	```
	{
		"folder_name":string,
		"parent_id":string(Optional)
	}
	```
		- folder_name is the name of the new folder
		- parent_id is the id of the parent folder,if none given the folder is created in user's home directory
	- Headers
		- Authorization: \<access_token> (access_token in string format)
	- Response
		- Status Code:200
		- Data(JSON)
		```
		{
			'id': string, 
			'kind': string,
			'mimeType': string,
			'name': string
		}
		```
	- Python Example
		```
		url = f"{baseURL}/files/folder"
		try:
			f = open("token.json", "r")
			access_token = f.read()
			f.close()
		except:
			print("Token not found")
			return
		#print("Access token is:",access_token)
		headers = {'Authorization': access_token}
		if parent_id:
			payload = {'folder_name': folder_name , "parent_id": parent_id }
		else:
			payload = {'folder_name': folder_name}
		r = requests.post(url, json=payload,headers=headers)
		response = r.json()
		print(response)

		```

- POST /v1/files/file  
Endpoint to upload a file
	- Request Data(MultipartForm)
		- parent_id:string (Optional)
		- file:bytes -> Bytes of the file you want to upload
	- Headers
		- Authorization: \<access_token> (access_token in string format)
	-Response
		- Status Code:200
		- Data(JSON)
		```
		{
			'id': string, 
			'kind': string,
			'mimeType': string,
			'name': string
		}
		```
	- Python Example
		```
		url = f"{baseURL}/files/file"
		try:
			f = open("token.json", "r")
			access_token = f.read()
			f.close()
		except:
			print("Token not found")
			return
		#print("Access token is:",access_token)
		headers = {'Authorization': access_token}
		files = {'file': open(filepath, 'rb')}
		if parent_id:
			payload = {'parent_id': parent_id}
			r = requests.post(url, data=payload,files=files,headers=headers)
		else:
			r = requests.post(url,files=files,headers=headers)
		print(r.json())
		```

### Permission Endpoints
- POST /v1/permissions/permission  
Endpoint to add a permission to a file.A permission is the ability to share a file with other users
	- Request Data(JSON)
		```
		{
			file_id: string,
			role: string,
			type: string, 
			emails: []string
		}
		```
		role: The role granted by this permission.Valid values are:
        - owner
        - organizer
        - fileOrganizer
        - writer
        - commenter
        - reader

        type: The type of the grantee. Valid values are:
        - user
        - group
        - anyone  
        When creating a permission, if type is user or group, you must provide an emailAddress for the user or group
    - Headers
		- Authorization: \<access_token> (access_token in string format)

    - Response
		- Status Code:200
		- Data(JSON)  
	    	```
			{
				message: string
			}
			```
	- Python Example
		```
		url = f"{baseURL}/permissions/permission"
		try:
			f = open("token.json", "r")
			access_token = f.read()
			f.close()
		except:
			print("Token not found")
			return
		#print("Access token is:",access_token)
		headers = {'Authorization': access_token}
		emails = ["email1@gmail.com","email2@gmail.com"]
		payload = {
			'file_id': file_id,
			"role":"reader",
			"type":"user",
			"emails":emails
		}
		r = requests.post(url, json=payload,headers=headers)
		print(r.json())
		```

- GET /v1/permissions/\<file_id>  
	Endpoint to get the permissions of a file

	- Headers
		- Authorization: \<access_token> (access_token in string format)

	- Response
		- Status Code:200
		- Data(JSON)
		```
		{
			"Permissions":[
				{
					"emailAddress": string,
  					"role": string,
  					"type": string
				}
			]
		}
		```
	- Python Example
		```
		url = f"{baseURL}/permissions/{file_id}"
		try:
			f = open("token.json", "r")
			access_token = f.read()
			f.close()
		except:
			print("Token not found")
			return
		headers = {'Authorization': access_token}
		r = requests.get(url,headers=headers)
		if r.status_code == 200:
			permissions = r.json()['Permissions']
			for permission in permissions:
				print(json.dumps(permission,indent=2))
		else:
			print(r.json())	
		```