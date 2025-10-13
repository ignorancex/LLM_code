import requests
from bs4 import BeautifulSoup, NavigableString
import re
import csv
from datetime import datetime, timedelta
import pytz

# The following rows are not for the generator supply
excluded_first_col = {
    'COGENERATION', 'WIND', 'COMBINED CYCLE', 'GAS FIRED STEAM',
    'SOLAR', 'SIMPLE CYCLE', 'HYDRO', 'OTHER', 'ENERGY STORAGE', 'TOTAL'
}

url = 'http://ets.aeso.ca/ets_web/ip/Market/Reports/CSDReportServlet'
response = requests.get(url)
response.encoding = 'utf-8'
soup = BeautifulSoup(response.text, 'html.parser')

rows = soup.find_all('tr')
final_data = []

for row in rows:
    cols = row.find_all('td')
    if len(cols) != 4:
        continue

    clean_row = []
    valid = True
    for col in cols:
        if len(col.contents) != 1 or not isinstance(col.contents[0], NavigableString):
            valid = False
            break
        clean_row.append(col.get_text(strip=True))

    if not valid:
        continue

    generator = clean_row[0]
    if generator in excluded_first_col:
        continue

    match = re.search(r'\(([^)]+)\)', generator)
    if not match:
        raise ValueError(f"Can't find Generator Name：{generator}")

    generator_name = match.group(1)
    supply = clean_row[2]

    final_data.append([generator, generator_name, supply])

# Get Alberta current time
alberta_tz = pytz.timezone('Canada/Mountain')
now = datetime.now(alberta_tz)
timestamp_str = now.strftime('%Y%m%d_%H%M')

# Construct File name
filename = f'GeneratorSupply_{timestamp_str}.csv'

# Store into CSV
with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Generator', 'Generator Name', 'Supply'])
    writer.writerows(final_data)

print("Data has been written to", filename)