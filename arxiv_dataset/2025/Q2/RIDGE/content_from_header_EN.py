import openai
import os, sys
from tqdm import tqdm
from global_config import OPENAI_KEY


# Initialize the OpenAI API client
openai.api_key = OPENAI_KEY

# =================================================================================================
length = "200-300"

prompt = """\
Based on the provided form title, create realistic and diverse content for the paper form, including multiple key-value pairs. These pairs can be one-to-one, one-to-many, or multiple-choice questions. The more diversity, the better. Avoid using overly common names like "John Doe."
Use <h{num}></h{num}> to enclose each paragraph. Each item should start with the "-" symbol, and the number of "-" symbols should indicate the hierarchy level (e.g. "--" is related to the last "-" before it, "---" is related to the last "--" before it). Make sure the hierarchy level is correct.
Generate content of approximately {length} words, placing it within <content></content>. The output should be in plain text. DO NOT USE MARKDOWN, TAB, and ICON!
Pretend you are filling out this paper form, so include answers that appear realistic, diverse, and feature a complex hierarchical structure."""

example_q = "GREENWOOD SPRING FESTIVAL FEEDBACK FORM"

example_a = """\
<content>
- Event Date: April 22, 2023
- Report Date: May 10, 2023
- Team: Eastern - GSF
- Event Coordinator: A. R. Mitchell

<h1> Promotional Items:
- T-Shirts: 25
- Brochure: 75
- Banners: 8
- Stickers: 500
- Date Received: 04/15/23
</h1>

<h2> RSVP Details:
- # Invited: 150
- Method of RSVP:
-- Online Form ☑
-- Email ☑
-- Phone Call ☐
-- In-Person ☐
- Date Invitations Sent: 03/20/23
</h2>

<h3> Displays Setup:
- Booths: 5
- Outdoor Banners: 10
- Flyer Stands: 15
</h3>

<h4> Hospitality Area Review
- Food Quality
-- ☐ Bad
-- ☐ Average
-- ☑ Excellent
- Cleanliness
-- Dining Area
--- ☐ Bad
--- ☐ Average
--- ☑ Excellent
-- Waste Management
--- ☐ Bad
--- ☑ Average
--- ☐ Excellent
-- Restroom Facilities
--- ☐ Bad
--- ☐ Average
--- ☑ Excellent
- Service
-- Staff Friendliness
--- ☐ Bad
--- ☐ Average
--- ☑ Excellent
-- Efficiency
--- ☐ Bad
--- ☐ Average
--- ☑ Excellent
</h4>
</content>
"""
# =================================================================================================

input_file_path = f"headers/{sys.argv[1]}.txt"
output_dir = f"contents/{os.path.basename(input_file_path).split('.')[0]}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_file =  open(input_file_path, "r", encoding="utf8")
headers = input_file.read().splitlines()
num_file = len(headers)
input_file.close()


for file_id in tqdm(range(0, num_file)):
    header = headers[file_id]
    header = header.split(".")[-1].strip()
    print(header)

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(length=length, num="{num}")},
            {"role": "user", "content": example_q},
            {"role": "assistant", "content": example_a},
            {"role": "user", "content": header}
        ],
    )
    response = response.choices[0].message.content
    print(response)

    with open(f'{output_dir}/{file_id:06d}.txt', 'w', encoding="utf8") as f:
        f.write(header + '\n' + response)