import json

from judgetuning.jsonfix.json_fixer import JSONFixer


def test_jsonfixer():
    string = '{bestModel: B, "explanation": "Model B provides a more diverse and accurate set of countries.", "confidence": 0.8}'
    fixer = JSONFixer()
    fixed_string = fixer(string=string)
    assert JSONFixer.is_valid_json(fixed_string)
    json_output = json.loads(fixed_string)
    print(json_output)
    for key in ["bestModel", "explanation", "confidence"]:
        assert key in json_output


def test_jsonfixer2():
    string = (
        r'{"explanation":'
        r' "Model A provides more comprehensive information about the actors\' backgrounds and offers '
        r'more specific examples",  "scoreA": 0.83, "scoreB": 0.72, "confidence": 0.85'
    )
    fixer = JSONFixer()
    fixed_string = fixer(string=string)
    assert JSONFixer.is_valid_json(fixed_string)


if __name__ == "__main__":
    test_jsonfixer()
