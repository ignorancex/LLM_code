# coding: utf-8
import pickle
from files.scripts.analysis import *
from pathlib import Path
base = Path("/project2/rkessler/SURVEYS/DES/USERS/parmstrong/dev/bias_validation/")
output = base / "outputs"
wfits = output / "wfits.p"
wfits = pickle.load(open(wfits, "rb"))
gp = apply_GP(wfits)
fr = list(wfits["FR"].values())[0]
gp = get_GP(fr)
gp.predict(np.array([0.19]).reshape(-1, 1))
