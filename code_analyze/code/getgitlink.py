import fitz
import re


def extract_github_links(pdf_path):
    doc = fitz.open(pdf_path)
    github_links = set()
    pattern = re.compile(r"https?://github\.com/[^\s,)]+")

    for page in doc:
        text = page.get_text()
        github_links.update(pattern.findall(text))

    return list(github_links)


pdf_file = "1905.00075.pdf"
links = extract_github_links(pdf_file)
print(links)
