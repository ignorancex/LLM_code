import requests
import json

# Define the prediction endpoint
url = "http://127.0.0.1:12500/predict"

# Test cases with the correct data format based on the TSV sample
test_cases = [
    {
        "fiber_radius": 20.0,
        "helix_radius": 50.0,
        "n_turns": 10.0,
        "pitch": 300.0,
    }
]

for i, test_case in enumerate(test_cases, 1):
    # Make the POST request
    try:
        response = requests.post(
            url,
            data=json.dumps(test_case),
            headers={"Content-type": "application/json"},
            timeout=10,
        )

        # Check if request was successful
        if response:
            result = response.json()
            if result["success"]:
                print(result)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
