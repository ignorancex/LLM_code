#mcandrew

import numpy as np
import pandas as pd

if __name__ == "__main__":

    stage1Reviews = pd.read_csv('fromWoS_2_AgreedArticles/agreedOnArticles_20190325.csv')
    disagrees = stage1Reviews[stage1Reviews['agree']==0]

    revs = ['ngr','nw','gcg']
    newDisagreesForReview = pd.DataFrame()
    for rev in revs:
        potentialRevs = list(set(revs) - set([rev]))
        revArticles = disagrees[disagrees.reviewer==rev].reset_index()

        rows = np.arange(len(revArticles))
        rev1 = np.random.choice(rows,size=int(len(rows)/2.),replace=False)
        rev2 = list(set(rows) - set(rev1))

        revArticles.loc[revArticles.index.isin(rev1),'tieBreaker'] = potentialRevs[0]
        revArticles.loc[revArticles.index.isin(rev2),'tieBreaker'] = potentialRevs[1]

        newDisagreesForReview = newDisagreesForReview.append(revArticles)
    newDisagreesForReview.to_csv('fullDataDisagree_20190325.csv')
    newDisagreesForReview[['TI','AB','DI','PY','tieBreaker']].to_csv('subsetDataDisagree_20190325.csv')

    gcgTieBreaker = newDisagreesForReview.loc[newDisagreesForReview.tieBreaker=='gcg',['TI','AB','DI','PY']]
    gcgTieBreaker.to_csv('gcgTieBreaker__20190325.csv')

    nwTieBreaker  = newDisagreesForReview.loc[newDisagreesForReview.tieBreaker=='nw',['TI','AB','DI','PY']]
    nwTieBreaker.to_csv('nwTieBreaker__20190325.csv')
    
    ngrTieBreaker = newDisagreesForReview.loc[newDisagreesForReview.tieBreaker=='ngr',['TI','AB','DI','PY']]
    ngrTieBreaker.to_csv('ngrTieBreaker__20190325.csv')
