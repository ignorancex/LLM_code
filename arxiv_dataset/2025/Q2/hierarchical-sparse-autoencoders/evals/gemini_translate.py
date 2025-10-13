import llm
import json
import asyncio
from time import sleep

async def main():
    with open("english.json", "r", encoding="utf-8") as file:
        translations = json.load(file)

    model = llm.get_async_model("gemini-2.0-flash")
    for lang, code in [("french", "fr"), ("spanish", "es"), ("german", "de")]:
        try:
            schema = llm.schema_dsl(f"{code}: translated text")
            responses = []
            for item in translations:
                prompt = f"Translate the following text into {lang}: {item["en"]}"
                responses.append(model.prompt(prompt, schema=schema))
            responses = await asyncio.gather(*responses)
            [t.update(json.loads(await r.text())) for t, r in zip(translations, responses)]
            sleep(60) # rate limit
        except Exception as e:
            print(f"Error for {lang}: {e}")
            break

    with open("translations.json", 'w', encoding='utf-8') as file:
        json.dump(translations, file, indent=2, ensure_ascii=False)

asyncio.run(main())
