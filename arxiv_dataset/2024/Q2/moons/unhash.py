import rebound

def unhash(h,names):
    
    hashes = {n:rebound.hash(n) for n in names}
    
    return [key for key, value in hashes.items() if value.value == h.value][0]