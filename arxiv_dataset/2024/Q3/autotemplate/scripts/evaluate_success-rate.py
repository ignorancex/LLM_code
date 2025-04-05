import json
import re
from collections import defaultdict

import fire
import pandas as pd

pd.options.display.float_format = '{:,.2f}'.format


def run(hyp_file, ref_file):
    sr = defaultdict(list)
    for hyp, ref in zip(open(hyp_file), open(ref_file)):
        ref = json.loads(ref)
        keywords = re.split(r"<extra_id_\d*>", ref["src"].split(" | ", maxsplit=1)[0][6:])
        keywords = {w.strip() for w in keywords if w.strip()}
        sr["sr"].append(all(w in hyp for w in keywords))

    scores = defaultdict(list)
    for key in sr:
        scores["key"].append(key)
        scores["Data size"].append(len(sr[key]))
        scores["Num Success"].append(sum(sr[key]))
        scores["Success Rate"].append(100 * sum(sr[key]) / len(sr[key]))

    print(pd.DataFrame(scores, ))


if __name__ == '__main__':
    fire.Fire(run)
