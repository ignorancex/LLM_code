import json


def jprint(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))
