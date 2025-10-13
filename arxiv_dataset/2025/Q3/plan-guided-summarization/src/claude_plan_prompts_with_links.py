def prompt_icl_with_links_1(text):
    prompt = """Human: Given a text, devise a plan for a summary, based on major events in the text. Each plan point should be a simple sentence that contains between 3-10 words. Here are some example text plan pairs:

Each sentence in the Text ends with the sentence number such as [1], [2] etc.

Text:

Christina Aguilera had a good Valentine's Day: She announced yesterday that she's engaged to Matt Rutler, her boyfriend of three years, Radar reports. [1] Aguilera, 33, posted a photo to Facebook of her and Rutler holding hands on the beach—and her hand is sporting a huge ring. [2] "He asked and I said..." reads the caption. [3] Aguilera was previously married to Jordan Bratman, with whom she has a 6-year-old son. [4] (As they say, the couple who gets arrested together stays together.) [5]

Plan:
1. Christina Aguilera had a good Valentine's Day. [1]
2. She announced her engagement to Matt Rutler. [1]
3. She posted a photo of herself and Rutler holding hands. [2]
4. She was previously married to Jordan Bratman. [4]
5. She has a son with Bratman. [4]
6. The couple was arrested together previously. [5]

Text:

The US stands by the "one-China" policy, but that doesn't mean it can't sell weapons directly to Taiwan, citing the Taiwan Relations Act to ensure Taiwan can adequately defend itself—and China isn't happy about it. [1] The Obama administration announced a $1.8 billion arms package sale to Congress on Wednesday, Reuters reports, including guided-missile frigates, anti-tank missiles, Amphibious Assault Vehicles, and $416 million worth of guns, ammo, and other supplies. [2] The announcement came amid reports that the US had stalled the sale to avoid hearing about it from China, which still claims Taiwan as a territory, per the Wall Street Journal. [3] Reuters notes the sale comes as US-China relations simmer over the latter's man-made islands in the South China Sea and US patrols in those waters. [4] China notes it's going to sanction the companies involved in the sale (including Lockheed Martin and Raytheon), with a foreign ministry official telling Xinhua that the sale flouts international rules and "severely" damages China's sovereignty. [5] "China's government and companies will not carry out cooperation and commercial dealings with these types of companies," a ministry spokesman says. [6] A Pentagon spokesman gave the equivalent of an eyeroll Wednesday, per the New York Times, noting, "The Chinese can react to this as they see fit. [7] It's a clear-eyed, sober view of an assessment of Taiwan's defense needs. [8] There's no need for it to have any derogatory effect on our relationship with China." [9] Meanwhile, the AP notes that China has issued similar threats before, with "no evidence they've had any meaningful effect." (All this despite a lengthy handshake last month.) [10]

Plan:
1. US announced an arms package sale to Taiwan. [2]
2. China is not happy about it. [1]
3. China threatens to sanction companies involved in the sale. [5]
4. US shrugs off the threat. [9]
5. China has issued similar threats before without any meaningful effect. [10]

Text:

Princess Ayako of Japan on Monday wed Kei Moriya at Tokyo's Meiji Shrine and became Princess Ayako no more. [1] Per imperial law, in choosing not to marry a royal, the 28-year-old had to give up her royal status. [2] She'll be given a parting sum of $950,000 from the Japanese government, CNN reports. [3] Ayako is the daughter of the emperor's cousin, and Moriya, 32, works for major shipping company Nippon Yusen. [4] She bid farewell to Emperor Akihito last week. [5] After emerging from the shrine, she said, "I will leave the imperial family today, but I will remain unchanged in my support for his majesty and her majesty." [6] The AP describes their wedding as a "ritual-filled ceremony" that involved sharing a cup of sake. [7] She wore a Heian-era style hairdo, which is swept back into a ponytail, and a traditional robe splashed with red and green patterns, while Moriya wore coattails. [8] She later changed into a red Japanese robe. [9] "I am filled with happiness," she told reporters after the ceremony. [10] Ayako and Moriya met because their mothers were friends and they hit it off immediately. [11] Moriya said he had been attracted to her gentle spirit. [12] "And I knew I wanted to spend the rest of my life with her," he said. (The eldest granddaughter of the emperor is set to marry a commoner in 2020, and it's been a bit of a bumpy ride.) [13]

Plan:
1. Princess Ayako of Japan married Kei Moriya. [1]
2. She lost her royal status. [2]
3. She received money from the Japanese government. [3]
4. Kei Moriya works for a major shipping company. [4]
5. Princess Ayako bid farewell to Emperor Akihito. [5]
6. She said her support for the emperor and empress is unchanged. [6]
7. The couple had a traditional wedding ceremony. [7, 8, 9]

Assistant: Ok. How many plan points do you want me to include?

Human: This will depend on the length of the text. If the text is long you can include many plan points. Make sure each significant event or occurrence is represented in the plan.

Assistant: What are the numbers in the bracket such as [1], [2] etc at the end of each plan point?

Human: Good question. For each plan point, you are required to cite the relevant sentence number which can be used to validate the information contained in the plan point. If multiple sentences need to be cited, then separate the sentence numbers with comma such as [1, 2, 3] or [8, 10].

Assistant: Ok, so the numbers at the end of the plan point correspond to the relevant sentence numbers based on which the plan point was generated.

Human: Yes, that is correct. Please be very careful with the citation. It is very important that you get the citation correct for all of the plan points. Please note again that each sentence in the Text ends with the sentence number such as [1], [2] etc.

Assistant: Ok, I will do my best.

Human: Now it's your turn to write plans. I'll give you the text and you give me the plan with citations for each plan point. Provide your response after "Plan:".

Text:
{}

Assistant:""".format(
        text
    )
    return prompt


def prompt_icl_with_links_1a(text):
    prompt = """Human: Given a text, with enumerated sentences, devise a plan for a summary based on major events in the text. Each plan point should be a simple, short sentence (3-10 words), followed by references to all relevant sentences that can be used to validate the information it contains.

    Here are some examples:

<example>
Text:

Christina Aguilera had a good Valentine's Day: She announced yesterday that she's engaged to Matt Rutler, her boyfriend of three years, Radar reports. [1] Aguilera, 33, posted a photo to Facebook of her and Rutler holding hands on the beach—and her hand is sporting a huge ring. [2] "He asked and I said..." reads the caption. [3] Aguilera was previously married to Jordan Bratman, with whom she has a 6-year-old son. [4] (As they say, the couple who gets arrested together stays together.) [5]

Plan:
1. Christina Aguilera had a good Valentine's Day. [1]
2. She announced her engagement to Matt Rutler. [1]
3. She posted a photo of herself and Rutler holding hands. [2]
4. She was previously married to Jordan Bratman. [4]
5. She has a son with Bratman. [4]
6. The couple was arrested together previously. [5]
</example>

<example>
Text:

The US stands by the "one-China" policy, but that doesn't mean it can't sell weapons directly to Taiwan, citing the Taiwan Relations Act to ensure Taiwan can adequately defend itself—and China isn't happy about it. [1] The Obama administration announced a $1.8 billion arms package sale to Congress on Wednesday, Reuters reports, including guided-missile frigates, anti-tank missiles, Amphibious Assault Vehicles, and $416 million worth of guns, ammo, and other supplies. [2] The announcement came amid reports that the US had stalled the sale to avoid hearing about it from China, which still claims Taiwan as a territory, per the Wall Street Journal. [3] Reuters notes the sale comes as US-China relations simmer over the latter's man-made islands in the South China Sea and US patrols in those waters. [4] China notes it's going to sanction the companies involved in the sale (including Lockheed Martin and Raytheon), with a foreign ministry official telling Xinhua that the sale flouts international rules and "severely" damages China's sovereignty. [5] "China's government and companies will not carry out cooperation and commercial dealings with these types of companies," a ministry spokesman says. [6] A Pentagon spokesman gave the equivalent of an eyeroll Wednesday, per the New York Times, noting, "The Chinese can react to this as they see fit. [7] It's a clear-eyed, sober view of an assessment of Taiwan's defense needs. [8] There's no need for it to have any derogatory effect on our relationship with China." [9] Meanwhile, the AP notes that China has issued similar threats before, with "no evidence they've had any meaningful effect." (All this despite a lengthy handshake last month.) [10]

Plan:
1. US announced an arms package sale to Taiwan. [2]
2. China is not happy about it. [1]
3. China threatens to sanction companies involved in the sale. [5]
4. US shrugs off the threat. [9]
5. China has issued similar threats before without any meaningful effect. [10]
</example>

<example>
Text:

Princess Ayako of Japan on Monday wed Kei Moriya at Tokyo's Meiji Shrine and became Princess Ayako no more. [1] Per imperial law, in choosing not to marry a royal, the 28-year-old had to give up her royal status. [2] She'll be given a parting sum of $950,000 from the Japanese government, CNN reports. [3] Ayako is the daughter of the emperor's cousin, and Moriya, 32, works for major shipping company Nippon Yusen. [4] She bid farewell to Emperor Akihito last week. [5] After emerging from the shrine, she said, "I will leave the imperial family today, but I will remain unchanged in my support for his majesty and her majesty." [6] The AP describes their wedding as a "ritual-filled ceremony" that involved sharing a cup of sake. [7] She wore a Heian-era style hairdo, which is swept back into a ponytail, and a traditional robe splashed with red and green patterns, while Moriya wore coattails. [8] She later changed into a red Japanese robe. [9] "I am filled with happiness," she told reporters after the ceremony. [10] Ayako and Moriya met because their mothers were friends and they hit it off immediately. [11] Moriya said he had been attracted to her gentle spirit. [12] "And I knew I wanted to spend the rest of my life with her," he said. (The eldest granddaughter of the emperor is set to marry a commoner in 2020, and it's been a bit of a bumpy ride.) [13]

Plan:
1. Princess Ayako of Japan married Kei Moriya. [1]
2. She lost her royal status. [2]
3. She received money from the Japanese government. [3]
4. Kei Moriya works for a major shipping company. [4]
5. Princess Ayako bid farewell to Emperor Akihito. [5]
6. She said her support for the emperor and empress is unchanged. [6]
7. The couple had a traditional wedding ceremony. [7, 8, 9]
</example>

Assistant: Ok. How many plan points do you want me to include?

Human: This will depend on the length of the text. If the text is long you can include many plan points. Make sure each significant event or occurrence is represented in the plan.

Assistant: What are the numbers in the bracket such as [1], [2] etc at the end of each plan point?

Human: Good question. For each plan point, you are required to cite the relevant sentence number which can be used to validate the information contained in the plan point. If multiple sentences need to be cited, then separate the sentence numbers with comma such as [1, 2, 3] or [8, 10].

Assistant: Ok, so the numbers at the end of the plan point correspond to the relevant sentence numbers based on which the plan point was generated.

Human: Yes, that is correct. Please be very careful with the citation. It is very important that you get the citation correct for all of the plan points. Please note again that each sentence in the Text ends with the sentence number such as [1], [2] etc.

Assistant: Ok, I will do my best.

Human: Now it's your turn to write plans. I'll give you the text and you give me the plan with citations for each plan point. Provide your response after "Plan:".

Text:
{}

Assistant:""".format(
        text
    )
    return prompt


def prompt_icl_with_links_2(text):
    prompt = """Human: Given a text, with enumerated sentences, devise a plan for a summary based on major events in the text. Each plan point should be a simple, short sentence, followed by references to all relevant sentences that can be used to validate information it contains.

    Here are some examples:

<example>
Text:

Christina Aguilera had a good Valentine's Day: She announced yesterday that she's engaged to Matt Rutler, her boyfriend of three years, Radar reports. [1] Aguilera, 33, posted a photo to Facebook of her and Rutler holding hands on the beach—and her hand is sporting a huge ring. [2] "He asked and I said..." reads the caption. [3] Aguilera was previously married to Jordan Bratman, with whom she has a 6-year-old son. [4] (As they say, the couple who gets arrested together stays together.) [5]

Plan:
1. Christina Aguilera had a good Valentine's Day. [1]
2. She announced her engagement to Matt Rutler. [1]
3. She posted a photo of herself and Rutler holding hands. [2]
4. She was previously married to Jordan Bratman. [4]
5. She has a son with Bratman. [4]
6. The couple was arrested together previously. [5]
</example>

<example>
Text:

The US stands by the "one-China" policy, but that doesn't mean it can't sell weapons directly to Taiwan, citing the Taiwan Relations Act to ensure Taiwan can adequately defend itself—and China isn't happy about it. [1] The Obama administration announced a $1.8 billion arms package sale to Congress on Wednesday, Reuters reports, including guided-missile frigates, anti-tank missiles, Amphibious Assault Vehicles, and $416 million worth of guns, ammo, and other supplies. [2] The announcement came amid reports that the US had stalled the sale to avoid hearing about it from China, which still claims Taiwan as a territory, per the Wall Street Journal. [3] Reuters notes the sale comes as US-China relations simmer over the latter's man-made islands in the South China Sea and US patrols in those waters. [4] China notes it's going to sanction the companies involved in the sale (including Lockheed Martin and Raytheon), with a foreign ministry official telling Xinhua that the sale flouts international rules and "severely" damages China's sovereignty. [5] "China's government and companies will not carry out cooperation and commercial dealings with these types of companies," a ministry spokesman says. [6] A Pentagon spokesman gave the equivalent of an eyeroll Wednesday, per the New York Times, noting, "The Chinese can react to this as they see fit. [7] It's a clear-eyed, sober view of an assessment of Taiwan's defense needs. [8] There's no need for it to have any derogatory effect on our relationship with China." [9] Meanwhile, the AP notes that China has issued similar threats before, with "no evidence they've had any meaningful effect." (All this despite a lengthy handshake last month.) [10]

Plan:
1. US announced an arms package sale to Taiwan. [2]
2. China is not happy about it. [1]
3. China threatens to sanction companies involved in the sale. [5]
4. US shrugs off the threat. [9]
5. China has issued similar threats before without any meaningful effect. [10]
</example>

<example>
Text:

Princess Ayako of Japan on Monday wed Kei Moriya at Tokyo's Meiji Shrine and became Princess Ayako no more. [1] Per imperial law, in choosing not to marry a royal, the 28-year-old had to give up her royal status. [2] She'll be given a parting sum of $950,000 from the Japanese government, CNN reports. [3] Ayako is the daughter of the emperor's cousin, and Moriya, 32, works for major shipping company Nippon Yusen. [4] She bid farewell to Emperor Akihito last week. [5] After emerging from the shrine, she said, "I will leave the imperial family today, but I will remain unchanged in my support for his majesty and her majesty." [6] The AP describes their wedding as a "ritual-filled ceremony" that involved sharing a cup of sake. [7] She wore a Heian-era style hairdo, which is swept back into a ponytail, and a traditional robe splashed with red and green patterns, while Moriya wore coattails. [8] She later changed into a red Japanese robe. [9] "I am filled with happiness," she told reporters after the ceremony. [10] Ayako and Moriya met because their mothers were friends and they hit it off immediately. [11] Moriya said he had been attracted to her gentle spirit. [12] "And I knew I wanted to spend the rest of my life with her," he said. (The eldest granddaughter of the emperor is set to marry a commoner in 2020, and it's been a bit of a bumpy ride.) [13]

Plan:
1. Princess Ayako of Japan married Kei Moriya. [1]
2. She lost her royal status. [2]
3. She received money from the Japanese government. [3]
4. Kei Moriya works for a major shipping company. [4]
5. Princess Ayako bid farewell to Emperor Akihito. [5]
6. She said her support for the emperor and empress is unchanged. [6]
7. The couple had a traditional wedding ceremony. [7-9]
</example>

Assistant: Ok. How many plan points do you want me to include?

Human: This will depend on the length of the text. If the text is long you can include many plan points. Make sure each significant event or occurrence is represented in the plan.

Assistant: Ok, sounds good.

Human: Now it's your turn to write plans. I'll give you the enumerated sentences and you give me the plan with citations for each plan point. Begin your response with "Here is a proposed plan for summarizing the key events in the text".

Text:
{}

Assistant:""".format(
        text
    )
    return prompt
