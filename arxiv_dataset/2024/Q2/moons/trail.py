import numpy as np

class Trail:
    
    def __init__(self,host='HOST',time=np.nan,parents=['PARENT1','PARENT2'],a=np.nan,e=np.nan,I=np.nan,Om=np.nan):
        
        self.host = host
        self.time = time
        self.parents = parents
        self.a = a
        self.e = e
        self.I = I
        self.Om = Om