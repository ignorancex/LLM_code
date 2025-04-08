#!/usr/bin/python

import math
import numpy as np

def Product_outer(a,b): return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def Product_inner(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def Sum(a,b): return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]
def Minus(a,b): return [b[0]-a[0],b[1]-a[1],b[2]-a[2]]
def Norm(a): return (a[0]**2+a[1]**2+a[2]**2)**0.5
def UnitVector(a):
    norm = Norm(a)
    return [a[0]/norm,a[1]/norm,a[2]/norm]

#p: a point on plan, n: normal to plane, point out of plane
def Distance_PointnPlane(p,n,x):
    d = -(n[0]*p[0] + n[1]*p[1]+n[2]*p[2])
    dis = abs(n[0]*x[0] + n[1]*x[1] + n[2]*x[2] + d)/(n[0]**2+n[1]**2+n[2]**2)**0.5
    return dis

#P: a point, V1,V2 = vectors defining the line
def Distance_PointnLine(P,V1,V2):
    return Norm(Product_outer(Minus(V1,P),Minus(V2,P)))/Norm(Minus(V1,V2))

def distance(a,b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

#PV:plane vector, V: vector
def Angle_PnV(PV1,PV2,V):
    normalV = Product_outer(PV2,PV1) 
    return abs(math.pi/2-Angle_VnV(normalV,V))

def ScalarProduct(m,v):
    for i in range(len(v)):
        v[i] *= m
    return v
#def ProjectionVectorOntoPlane(n,v):return Product_outer(n,Product_outer(v,n))
def ProjectionVectorOntoPlane(n,u): return Minus(ProjectionVectorOntoVector(n,u),u)
def ProjectionVectorOntoVector(v,u):
    unit_v = UnitVector(v)
    return ScalarProduct(Product_inner(u,unit_v),unit_v)

def ProjectionPointOntoVector(p1,p2,p):
    v = UnitVector(Minus(p1,p2))
    v2 = Minus(p1,p)
    l = Norm(v2)*Product_inner(v,v2)/Norm(v2)/Norm(v)
    #l = Product_inner(v,v2)/Norm(v)
    return [p1[0]+v[0]*l,p1[1]+v[1]*l,p1[2]+v[2]*l]

def Angle_VnV(A,B):
        import math
        normA = Norm(A)
        normB = Norm(B)
        
        AB = Product_inner(A,B)
        cos = AB/normA/normB
        if cos < -1.0:
            cos = -1
        if cos > 1:
            cos = 1
        acos = math.acos(cos)
        return acos

def Translation(coords,v,PN=1):
    Tcoords = []
    for c in coords:
        Tcoords.append([c[0]-v[0]*PN,c[1]-v[1]*PN,c[2]-v[2]*PN])
    return Tcoords

def RotationToAxis(coors,theta):
    rM = eulerAnglesToRotationMatrix(theta)
    #print (coors)
    #print (rM)
    return np.dot(coors,rM)

#https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,                  0,                   0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),  math.cos(theta[0])]
                    ])

    R_y = np.array([[ math.cos(theta[1]), 0, math.sin(theta[1])],
                    [                  0, 1,                  0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),  math.cos(theta[2]), 0],
                    [                 0,                   0, 1]
                    ])
                      
    R = np.dot(R_z, np.dot(R_y, R_x ))
 
    return R
    
#http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
def Rotation(coors,p1,dv,theta,fix=[]): #rotating the coors about the line through p1 with direction vector dv by the angle theta.
    import math
    dv = UnitVector(dv)
    cos = math.cos(theta); sin = math.sin(theta)
    a = p1[0];b = p1[1];c = p1[2]
    u = dv[0];v = dv[1];w = dv[2]
    u2 = u*u; v2 = v*v; w2 = w*w
    Rcoors = []
    for i in range(len(coors)):
        if i not in fix:
            x = coors[i][0];y = coors[i][1];z = coors[i][2]

            Rxyz = [(a*(v2+w2)-u*(b*v + c*w - u*x - v*y - w*z))*(1-cos) + x*cos + (-c*v + b*w - w*y + v*z)*sin,
                    (b*(u2+w2)-v*(a*u + c*w - u*x - v*y - w*z))*(1-cos) + y*cos + (c*u - a*w + w*x - u*z)*sin,
                    (c*(u2+v2)-w*(a*u + b*v - u*x - v*y - w*z))*(1-cos) + z*cos + (-b*u + a*v - v*x + u*y)*sin
                ]
            Rcoors.append(Rxyz)
        else:
            Rcoors.append(coors[i])
    return Rcoors

def Align(coor,std,vector):
    import math

    Tcoor = Translation(coor,vector[0])
    
    A = Minus(vector[0],vector[1])
    unitA = UnitVector(A)
    
    B = Minus(std[0],std[1])
    unitB = UnitVector(B)
    
    u = UnitVector(Product_outer(unitA,unitB)) #rotation axis, vertical to A and B
    theta = math.acos(Product_inner(unitA,unitB))
    
    return Rotation(Tcoor,u,theta)

def torsion(a1,a2,a3,a4) :
    b1 = Minus(a1,a2); b2 = Minus(a2,a3); b3 = Minus(a3,a4)
    b1b2 = Product_outer(b1,b2);b2b3 = Product_outer(b2,b3)
    theta = math.atan2(Product_inner(Product_outer(b1b2,b2b3),UnitVector(b2)),Product_inner(b1b2,b2b3))
    return abs(theta)

#https://github.com/ben-albrecht/qcl/blob/master/qcl/ccdata_xyz.py#L208
def internal2cartesian(dis,ang,tor,a,b,c): #a = distance atom, b = angle atom, c = dihedral atom. x-a-b-c
    """Calculate position of another atom based on internal coordinates"""
    v1 = Minus(b,a)
    v2 = Minus(c,a)

    n = Product_outer(v1, v2)
    nn = Product_outer(v1, n)

    n = UnitVector(n)
    nn = UnitVector(nn)

    n = ScalarProduct(-1*math.sin(tor),n)
    nn = ScalarProduct(math.cos(tor),nn)

    v3 = Sum(n,nn)
    v3 = UnitVector(v3)
    v3 = ScalarProduct(dis * math.sin(ang), v3)

    v1 = UnitVector(v1)
    v1 = ScalarProduct(dis * math.cos(ang), v1)

    position = Minus(v1, Sum(a, v3))

    return position

if __name__ == "__main__":
    None
