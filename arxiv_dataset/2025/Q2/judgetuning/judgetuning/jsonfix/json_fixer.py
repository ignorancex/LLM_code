import json
import logging
from json import JSONDecodeError
from pathlib import Path
import yaml
from json_repair import repair_json
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from judgetuning.utils import read_and_format


class JSONFixer:
    def __init__(self, client=None, model: str = None):
        self.client = client
        self.model = model

    @classmethod
    def is_valid_json(cls, string: str):
        try:
            json.loads(string)
            return True
        except JSONDecodeError:
            return False

    def __call__(self, string: str) -> str:
        try:
            string = repair_json(string, skip_json_loads=True)
        except RecursionError:
            return string
        if not self.is_valid_json(string):
            print(f"Could not fix json in input: {string}")
        return string
