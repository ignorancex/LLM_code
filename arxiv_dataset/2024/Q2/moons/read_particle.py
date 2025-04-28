def read_particle(string):
    
    chunks = string.split()
    
    for c in chunks:
        if "m=" in c:
            m = float(c[2:])
        if "x=" in c:
            x = float(c[2:])
        if "y=" in c:
            y = float(c[2:])
        if "z=" in c:
            z = float(c[2:])
        if "vx=" in c:
            vx = float(c[3:])
        if "vy=" in c:
            vy = float(c[3:])
        if "vz=" in c:
            vz = float(c[3:])
            
        return (m,x,y,z,vx,vy,vz)