import sys, os, json, numpy as np

def at(d, k):
    if (fp := k.find('.')) != -1:
        k, rk = k[:fp], k[fp+1:]
        if not isinstance(d, dict) or k not in d:
            return None
        return at(d[k], rk)
    return d[k] if k in d else None

path, crit, test_key = sys.argv[1:4]

crit = eval(crit)
path = os.path.expanduser(path)
ret = []

for d in os.listdir(path):
    d = os.path.join(path, d)
    if os.path.isdir(d):
        with open(os.path.join(d, 'hps.txt')) as fin:
            hps = json.load(fin)
        if any(at(hps, k) != v for k, v in crit.items()):
            continue
        with open(os.path.join(d, 'stdout')) as fin:
            try:
                log = eval(fin.readlines()[-1])
            except:
                sys.stderr.write(f'error loading {d}. experiment crashed?\n')
                continue
        ret.append(at(log, test_key))

print(len(ret), np.mean(ret), np.std(ret), np.percentile(ret, [50, 25, 75]))
