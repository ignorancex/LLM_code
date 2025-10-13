import google.generativeai as genai

_GEMINI_KEY = open('/Users/user/Desktop/gemini_key_fc_eval.txt', 'r').read()
_MAX_TOKENS = 3000
_TEMPERATURE = 0
_SEED = 10
_TIMEOUT = 180
_MODEL_NAME = "gemini-1.5-flash"

genai.configure(api_key=_GEMINI_KEY)


def _test():
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        "Tell me a story about a magic backpack.",
        generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        ),
    )
    print(response.text)


def main():
    _test()


if __name__ == '__main__':
    main()
