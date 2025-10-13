import requests
import json

print(
    requests.post(
        "http://127.0.0.1:12500/prediction_chembl",
        data=json.dumps({"smiles": "c1ccc2c(c1)c3cc4c(cc3C2)cc5c6cc7c8c(cc6)cccc8c9ccccc9C457"}),
        headers={"Content-type": "application/json"},
    ).json()
)


print("\n\n")
print(
    requests.post(
        "http://127.0.0.1:12500/prediction_supercon",
        data=json.dumps({"element": "Ba0.2La1.8Cu1O4-Y"}),
        headers={"Content-type": "application/json"},
    ).json()
)


print("\n\n")
print(
    requests.post(
        "http://127.0.0.1:12500/prediction_nanohelix",
        data=json.dumps(
            {"fiber_radius": 0.5, "helix_radius": 1.0, "n_turns": 6, "pitch": 120}
        ),
        headers={"Content-type": "application/json"},
    ).json()
)
