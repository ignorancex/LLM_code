import json
import os
import re

# Credit and based on https://github.com/cbaziotis/ekphrasis
from config import PREPROCESSING_REGEX


class ExManager:
    ext_path = os.path.join(PREPROCESSING_REGEX)

    with open(ext_path) as fh:
        expressions = json.load(fh)

    def get_compiled(self):
        regexes = {k.lower(): re.compile(self.expressions[k]) for k, v in
                   self.expressions.items()}
        return regexes

    def print_expressions(self):
        {print(k.lower(), ":", self.expressions[k])
         for k, v in sorted(self.expressions.items())}