import requests

arxiv_id = "1905.00075"
pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
pdf_path = f"{arxiv_id}.pdf"

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)

# links = extract_github_links(pdf_path)
# print(links)
