import sympy as sbl




x1, x2, z1, z2, z3, z4 = sbl.symbols("x1 x2 z1 z2 z3 z4")

x1c, x2c = sbl.symbols("x1c x2c")


L = sbl.Matrix([
 [ 0, 0,0, x1c],
 [ 0, 0, x2c,-1],
 [ 0, x2, -1, 0,],
 [x1, -1, 0, 0]])

Z = sbl.Matrix([
[z1, 0,0,0],
[0,z2,0,0],
[0,0,z3,0],
[0,0,0,z4]
])

Z_red = sbl.Matrix([
[z1, 0,0,0],
[0,0,0,0],
[0,0,0,0],
[0,0,0,0]
])


full = Z - L
print((full**-1)[0])


red = Z_red - L


print((red**-1)[0])


Yc = sbl.Matrix([
 [ 0, x1c],
 [  x2c,-1]
])


Y =  sbl.Matrix([
 [ 0, x2],
 [  x1,-1]
])


W = Yc*Y
Z = sbl.Matrix([
[z1, 0],
[0,z2],
])

print(W)

print( "det = {}".format( (Z-W).det() ))

print( ((Z-W)**-1)[0] )

print ( ((Z-W)**-1)[3]  )
