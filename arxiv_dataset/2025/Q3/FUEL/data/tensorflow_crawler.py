import re

import requests

urls = "https://www.tensorflow.org/api_docs/python/tf/raw_ops"


def crawler(url):
    response = requests.get(url)
    pattern = re.compile(
        r'<tr>\s*<td><a\s+id=([^ ]+)\s+href="([^"]+)">([^<]+)</a></td>\s*<td\s+style="[^"]*">([^<]+)</td>\s*<td\s+style="[^"]*">([^<]+)</td>\s*</tr>'
    )
    matches = pattern.findall(response.text)
    apis = []
    for match in matches:
        if match[-1] == "✔️":
            apis.append(match[-3])
    return apis


if __name__ == "__main__":
    with open("tensorflow_apis.txt", "w") as f:
        for api in crawler(urls):
            f.write(f"tf.raw_ops.{api}\n")
