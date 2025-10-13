import requests
import json

# For testing model accessing.
response = requests.post(
    "http://127.0.0.1:12500/chemlb35",
    data=json.dumps({"smiles": "c1ccc2c(c1)c3cc4c(cc3C2)cc5c6cc7c8c(cc6)cccc8c9ccccc9C457"}),
    headers={"Content-type": "application/json"},
)

print(response.text)

# # For testing document accessing.
# response = requests.post(
#     "http://127.0.0.1:12500/chemlb33_tutorial",
#     headers={'Content-type': 'application/json'}
# )
#
# print(response.text)
