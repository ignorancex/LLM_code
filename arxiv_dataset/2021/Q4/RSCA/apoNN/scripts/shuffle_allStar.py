import numpy as np
import apoNN.src.utils as apoUtils
import apogee.tools.read as apread
import apogee.tools.path as apogee_path
apogee_path.change_dr(16)

np.random.seed(0)
allStar = apread.allStar(exclude_star_bad=True)
shuffled_id = np.random.choice(len(allStar),len(allStar),replace=False)
print(len(allStar))
print(len(shuffled_id))
shuffled_allStar = allStar[shuffled_id]
apoUtils.dump(shuffled_allStar,"shuffled_allStar")
print("succesfully dumped shuffled_allStar")

