import open3d as o3d
import numpy as np
import math


"simple function to round stuff to make reporting better"
def roundPlanes(a,b,c):
    a = round(a/3, 2)
    b = round(b/3, 2)
    c = round(c/3, 2)
    equation = (f'{a}x + {b}y + {c}z')
    return(equation)

"given intial user selected points make a mesh triangle then convert to pcd to display along with original model"
def create_triangle_mesh(points, color=[1, 0, 0]):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    mesh.compute_triangle_normals()
    mesh.paint_uniform_color(color)
    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    
    return pcd

"data analysis function to solve for euler angles"
def get_euler_angles(planeA, planeB):
    # Normalize the plane normal vectors
    planeA = planeA / np.linalg.norm(planeA)
    planeB = planeB / np.linalg.norm(planeB)

    # Compute the axis of rotation
    # This works by finding the plane spanned by plane A and plane B
    # i.e., if we put the tail ends of the two vectors together, we find the plane that they make as the vectors extend to infinity
    # the cross product of then represents the normal vector of this plane, which is the axis we need to rotate by 
    axis = np.cross(planeA, planeB)
    axis = axis / np.linalg.norm(axis)

    # Compute the angle of rotation
    # dot product of two vectors is by definition equal to the cos of theta between them
    # dot product of planeA, planeB = cos(theta) --> rearrange to solve for theta
    angle = np.arccos(np.dot(planeA, planeB))

    # Skew-symmetric matrix  is a square matrix whose transpose equals its own negative. 
    # need this to solve rodrigyez formula to get rotation matrix    
    s_s_m = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Compute the rotation matrix R using rodriguez formula
    # I dont really understand the math behind this, but it works and is proven for like 200 years
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    r_mat = np.eye(3) + np.sin(angle) * s_s_m + (1 - np.cos(angle)) * np.dot(s_s_m, s_s_m)

    # Compute Euler angles using the rotation matrix
    # This is from ChatGPT  euler angles suffer from gimbal lock so computing the singluar is a way to prevent this (in the else statement, set z=0 as there are infinte rotations possible otherwise)
    # If not singular represents the mathetical way to convert rotation matrix to euler angles using zyx convention as is standard
    sy = np.sqrt(r_mat[0, 0] ** 2 + r_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(r_mat[2, 1], r_mat[2, 2])
        y = np.arctan2(-r_mat[2, 0], sy)
        z = np.arctan2(r_mat[1, 0], r_mat[0, 0])
    else:
        x = np.arctan2(-r_mat[1, 2], r_mat[1, 1])
        y = np.arctan2(-r_mat[2, 0], sy)
        z = 0

    # return the angles of interest and rotation matrix to later calculate the rotated plane
    return x, y, z, r_mat

"function generates equation of plane based on 3 points. Plane is a list of lists of xyz points. \""
def definePlaneEquation(plane, label): 
    p1 = np.array(plane[0])
    p2 = np.array(plane[1])
    p3 = np.array(plane[2])
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    cp = cp / np.linalg.norm(cp)  # Normalize the vector

    a, b, c = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)
    print('The equation of plane {4} is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d, label))

    return a,b,c


"Move whole structure to origin. if else statements allow us to decide which segment we are placing"
def moveToOrigin(data, label):
    x = label[0]
    y = label[1]
    z = label[2]
    out = []

    for planes in data:
        mid_temp = []
        for plane in planes:
            temp = []
            # for point in plane:
            new_x = plane[0] -x
            new_y = plane[1] -y
            new_z = plane[2] -z
            temp.append(new_x)
            temp.append(new_y)
            temp.append(new_z)
            mid_temp.append(temp)
        out.append(mid_temp)
    return out

"calculate angle to move points to z axis. Could probably combine with rotate_struct_1"
def getAngle_X(point):
    angle = math.atan2(point[2], point[1]) #is in radians
    angle = (-1*angle)
    return angle

"rotate any given points to x axis. x,y,z are lists of points, angle is the angle to rotate by "
def rotate_struct_1(points, angle):
    for plane in points:
        for value in plane:
            # Solve for new values using that angle and system of equations
            y = value[1] * math.cos(angle) - value[2] * math.sin(angle)
            z = value[1] * math.sin(angle) + value[2] * math.cos(angle)
            value[1] = y
            value[2] = z
    return points

"calculate angle to move points to z axis. Could probably combine with rotate_struct_2"
def getAngle_Z(point):
    angle = math.atan2(point[0], point[1]) #is in radians
    # angle = (-*angle)
    return angle

"rotate any given points to z axis. x,y,z are lists of points, angle is the angle to rotate by "
def rotate_struct_2(points, angle):
    for plane in points:
        for value in plane:
            x = value[0] * math.cos(angle) - value[1] * math.sin(angle) 
            y = value[1] * math.cos(angle) + value[0] * math.sin(angle)
            value[0] = x
            value[1] = y
    return points

"find angle between 2 input planes. Need to understand which angles we really \
care about and which ones come first "
def calcPlaneAngle(a1, b1, c1, a2, b2, c2):
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    angle = math.degrees(math.acos(d))
    print("Angle is " + str(angle) + " degree")
    return angle

