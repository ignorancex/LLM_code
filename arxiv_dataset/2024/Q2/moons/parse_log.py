import SimEvent

def parse_log(file):
    
    CEs = []
    colls = []
    ejecs = []
    
    try:
        with open(file,'r') as f:
            lines = f.readlines()
    except:
        print('Error reading '+file)
        return
    else:
        for i,l in enumerate(lines):
            if "CE between" in l:
                chunks = l.split()
                CEs.append(SimEvent('CE',(chunks[2],chunks[4]),float(chunks[-2])))

            if "collision" in l:
                chunks = l.split()
                colls.append(SimEvent('Coll',(chunks[0],chunks[5]),float(chunks[-2])))
                
        return (CEs, colls, ejecs)