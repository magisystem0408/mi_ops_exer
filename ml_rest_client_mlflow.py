import json
import requests

url = ""
headers = {'Content-Type': 'application/json'}

data = {
    "columns": ["age", "salary"],
    "data": [[40, 5000]]
}
request_data = json.dump(data)

response = requests.post(url, request_data, headers=headers)
print(response.text)
