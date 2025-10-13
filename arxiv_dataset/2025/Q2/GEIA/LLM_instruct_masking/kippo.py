messages=[
            {"role": "system", "content": f"{sys_prompt}"},
            {
                "role": "user",
                "content": 'Please identify and mask all sensitive entities in the following passage, leaving only one entity unmasked, in the following passage: " Barack Obama visited the United Nations headquarters in New York.". Please leave only one less informative/sensitive information related to an entity unmasked.',
            },
            {
                "role": "assistant",
                "content": "[[PERSON]] visited the United Nations headquarters in [[LOCATION]].",
            },
            {
                "role": "user",
                "content": 'Please identify and mask all sensitive entities in the following passage, leaving only one entity unmasked, in the following passage: " Elon Musk is the CEO of SpaceX, based in Hawthorne, California.". Please leave only one less informative/sensitive information related to an entity unmasked.',
            },
            {
                "role": "assistant",
                "content": "[[PERSON]] is the CEO of [[ORGANIZATION]], based in Hawthorne, California.",
            },
            {
                "role": "user",
                "content": 'Please identify and mask all sensitive entities in the following passage, leaving only one entity unmasked, in the following passage: " The CEO of Tesla, Elon Musk, met with the President of the United States.". Please leave only one less informative/sensitive information related to an entity unmasked.',
            },
            {
                "role": "assistant",
                "content": "The CEO of [[ORGANIZATION]], [[PERSON]], met with the President of the United States.",
            },
            {
                "role": "user",
                "content": f'Please identify and mask all sensitive entities in the following passage, leaving only one entity unmasked, in the following passage: " {prompt}". Please leave only one less informative/sensitive information related to an entity unmasked.',
            },
    ]