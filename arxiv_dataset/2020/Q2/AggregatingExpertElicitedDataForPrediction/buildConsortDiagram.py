#mcandrew

import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import seaborn as sns

if __name__ == "__main__":

    fig,ax = plt.subplots()

    allBoxes = []
    
    articlesInSearch = Rectangle((0.30,0.90-0.0125),0.40,0.10,alpha=0.10)
    ax.text(0.50,0.90,'Web Of Science Article Search\n (285)',ha='center',va='bottom')
    ax.add_patch(articlesInSearch)

    #line
    ax.plot([0.50,0.50],[0.90-0.0125,0.575],'k-')

    agree2Exclude = Rectangle((0.70,0.70),0.20,0.10,facecolor='red',alpha=0.10)
    ax.text(0.80,0.70+0.0125,'Excluded\n (235)',ha='center',va='bottom')
    ax.add_patch(agree2Exclude)

    ax.plot([0.70,0.50],[0.70+0.0125+0.0285]*2,'k-')
   
    allInclude = Rectangle((0.40,0.475),0.20,0.10,alpha=0.10)
    ax.text(0.50,0.475+0.0125,'Included\n (67)',ha='center',va='bottom')
    ax.add_patch(allInclude)
    
    excludeAfter = Rectangle((0.65,0.25),0.30,0.10,color='red',alpha=0.10)
    ax.text(0.80,0.25+0.0125,'Excluded after analysis\n (14)',ha='center',va='bottom')
    ax.add_patch(excludeAfter)

    ax.plot([0.50,0.65],[0.30,0.30],'k-')

    final = Rectangle((0.40,0.10),0.20,0.10,alpha=0.10)
    ax.text(0.50,0.10+0.0125,'Analysis set\n (53)',ha='center',va='bottom')
    ax.add_patch(final)

    ax.plot([0.50,0.50],[0.475,0.20],'k-')

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')

    ax.text(-0.10,0.75,'Search query = (expert* or human* or crowd*)\n NEAR judgement AND (forecast* or predict*)\n AND (combin* or assimilat*)')

    ax.text(-0.10,0.425,'50 articles agreed upon by both\nreviewers plus 17 articles via a 3rd reviewer')
    
    fig.set_tight_layout(True)
    plt.savefig('./_G/F1/consortDiagram.pdf')
    plt.close()
