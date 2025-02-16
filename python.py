import requests

url = "http://127.0.0.1:5000/predict"
headers = {'Content-Type': 'application/json'}

# Replace this list with the actual 30 feature values used during training
data = {"features": [0.5] * 30}  # 30 values instead of 6

response = requests.post(url, json=data, headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())

