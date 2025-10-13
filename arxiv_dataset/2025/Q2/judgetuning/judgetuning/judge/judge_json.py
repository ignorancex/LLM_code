import json

from judgetuning.judge import Judge
from judgetuning.judge.io.judge_alpaca_eval import JudgeAlpacaEval
from judgetuning.judge.io.judge_arena_hard import JudgeArenaHard
from judgetuning.judge.judge_length import JudgeLength
from judgetuning.judge.judge_with_options.judge_options import JudgeOptions


def judge_from_json_str(json_str: str) -> Judge:
    json_dict = json.loads(json_str)
    for key in ["judge_cls", "judge_kwargs"]:
        assert key in json_dict
    found = False
    for cls in [
        JudgeOptions,
        JudgeAlpacaEval,
        JudgeArenaHard,
        JudgeLength,
    ]:
        if cls.__name__ == json_dict["judge_cls"]:
            found = True
            break
    assert found, f'invalid class {json_dict["judge_cls"]}'
    return cls.from_json(json_str)
