import math
import numpy as np
from matplotlib import pyplot as plt
from CODE.group_actions import *

def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)


#########################################################################################
### Chain Complex #######################################################################
#########################################################################################


class cell_complex_2:
    def __init__(self, coords, edges, faces, tol=0.000000001):
        self.totcoords = coords
        self.edges = edges
        self.faces = faces
        self.tol = tol
        
    def num(self, dim):
        #The number of cells of the given dimension
        if dim == 2:return len(self.faces)
        elif dim == 1:return len(self.edges)
        elif dim == 0:return self.totcoords.shape[0]
        else:return 0
    
    def adj(self, dim):
        #Adjacency matrix between dim cells and dim-1 cells
        adj = np.zeros((self.num(dim-1), self.num(dim)), dtype = int)
        if dim == 2:
            # Orientations are given by the order of nodes along the face.
            for j in range(0, self.num(2)):  # Which face
                face = np.array(self.faces[j])  # converts list to array
                for i in range(0, self.num(1)):  # Which edge
                    first = np.where(self.edges[i][0] == face)
                    for k in first[0]:  # Index of where edge i has a tail in face
                        if self.edges[i][1] == face[int(k - 1) % face.shape[0]]:  # If head of edge i is behind in face
                            adj[i, j] -= 1
                        if self.edges[i][1] == face[int(k + 1) % face.shape[0]]:  # If head of edge i is ahead in face
                            adj[i, j] += 1
        elif dim==1:
            for j in range(0, self.num(1)):
                adj[self.edges[j][0], j] -= 1
                adj[self.edges[j][1], j] += 1
        return adj
    
    def edge_norm(self, index):
        #Returns normalized vector parallel to the edge
        edgenorm= self.totcoords[self.edges[index][1]] - self.totcoords[self.edges[index][0]]
        if abs(np.linalg.norm(edgenorm)) > self.tol:
            edgenorm = edgenorm / np.linalg.norm(edgenorm)
        return edgenorm
    
    def perp_edge_norm(self, index):
        rot_matx = np.array([[0,-1],[1,0]]) #90 degrees CCW
        return np.dot(rot_matx, self.edge_norm(index))

    def face_centroid(self, index):
        #Returns the centroid of the face
        center = [0,0]
        face = self.faces[index]
        for i in range(0, len(face)):
            center += self.totcoords[face[i]] / len(face)
        return center
    
    def edge_centroid(self, index):
        #Returns the centroid of the edge
        edge = self.edges[index]
        center = self.totcoords[edge[0]]/2 + self.totcoords[edge[1]]/2
        return center
    
    def face_orientation(self, index):
        #Returns True if face is oriented counter-clockwise
        #Otherwise returns False
        #The outer face is an exception. It's sign is flipped from normal
        if index == len(self.faces)-1:
            return True
        face = self.faces[index]
        summy = 0
        for i in range(0, len(face)):
            current_coord = self.totcoords[face[i]]
            mod_index = face[(i+1) % len(face)]
            next_coord = self.totcoords[mod_index]
            summy += (next_coord[0] - current_coord[0])*(next_coord[1] + current_coord[1])
        if summy <= 0:
            return True
        else:
            return False
    
    def plot(self):
        fig, ax = plt.subplots()
        for i in range(0, self.num(0)):
            X = self.totcoords[i][0]
            Y = self.totcoords[i][1]
            ax.text(X, Y, i)
            ax.scatter(X,Y)

        for i in range(0, self.num(1)):
            edge = self.edges[i]
            edgex = [self.totcoords[edge[0], 0], self.totcoords[edge[1], 0] ]
            edgey = [self.totcoords[edge[0], 1], self.totcoords[edge[1], 1] ]
            ax.plot(edgex, edgey, color = 'black')
        ax.set_title('Primal')
        plt.show()


#########################################################################################
### Wedge To Complex Toolbox ############################################################
#########################################################################################

def center_shift(value, cap, center = True):
    if center:
        return (value  % cap) + 1
    return value % cap

def wedge_to_complex(num_wedge, pol_wedge, E_wedge, F_wedge, out_list, in_list, M_wedge = {}):
    #This takes the local coordinates, edges, faces at a wedge
    # and extends it to a full rotationally/mirror symmetric cell complex
    len_wedge = pol_wedge.shape[0]
    center = False
    if len(in_list) == 0:
        center = True
    
    #Generate cartesian vertex coordinates (coord)
    pol_coords = np.copy(pol_wedge).astype(dtype = float)
    if center:
        coords = np.array([[0.0, 0.0]])
    else:
        coords = np.empty((0,2))
    for i in range(0, num_wedge):
        for j in range(0, len_wedge):
            vec = np.array([pol2cart(pol_coords[j,0], pol_coords[j,1])])
            coords = np.append(coords, vec, axis = 0)
            pol_coords[j,1] += 2*math.pi/num_wedge
    num_coords = coords.shape[0]
    
    #Generate edge connectivity dictionary (E)
    #The center node is designated (-1) index
    E = {}
    for i in range(0, num_wedge):
        for key in E_wedge:
            edge_wedge = E_wedge[key]
            E[len(E)] = [-1]*2
            for j in range(0, len(edge_wedge)):
                if edge_wedge[j]< -0.5: #Is the center node
                    E[len(E)-1][j] = 0
                else:
                    E[len(E)-1][j] = center_shift(edge_wedge[j] + i*len_wedge, len_wedge*num_wedge, center)
    
    #Generate the face connectivity dictionary (F)
    F = {}
    for i in range(0, num_wedge):
        for key in F_wedge:
            face_wedge = F_wedge[key]
            F[len(F)] = [-1]*len(face_wedge)
            for j in range(0, len(face_wedge)):
                if face_wedge[j] < -0.5: #Is the center node
                    F[len(F)-1][j] = 0
                else:
                    F[len(F)-1][j] = center_shift(face_wedge[j] + i*len_wedge, len_wedge*num_wedge, center)
            
    #Add the outside face
    face_out = [-1]*(len(out_list)*num_wedge)
    for i in range(0, num_wedge):
        for j in range(0, len(out_list)):
            face_out[j + i*len(out_list)] = center_shift(out_list[j] + i*len_wedge, len_wedge*num_wedge, center)
    F[len(F)] = face_out
    #Add the inside face
    if not center:
        face_in = [-1] * (len(in_list)*num_wedge)
        for i in range(0, num_wedge):
            for j in range(0, len(in_list)):
                face_in[j+i*len(in_list)] = center_shift(in_list[j] + i*len_wedge, len_wedge*num_wedge, center)
        F[len(F)] = face_in
    
    #Generate the rotation permutation dictionary (R_perm)
    R_perm = {}
    if center:
        R_perm[len(R_perm)] = 0
    for i in range(0, num_wedge):
        for j in range(0, len_wedge):
            R_perm[len(R_perm)] = center_shift(j + (i+1)*len_wedge, len_wedge*num_wedge, center)

    cell_complex = cell_complex_2(coords, E, F)
    if bool(M_wedge):
            #Generate the mirror permutation dictionary (M_perm)
        M_perm = {}
        if center:
            M_perm[len(M_perm)] = 0
        for i in range(0, num_wedge):
            M_num_wedge = num_wedge - i - 1
            for j in range(0, len_wedge):
                M_val = M_num_wedge*len_wedge + center + M_wedge[j] - 1
                M_perm[len(M_perm)] = center_shift(M_val, len_wedge*num_wedge, center)
        action = dihedral_group_action(coords, R_perm, num_wedge, M_perm, num_wedge)
    else:
        action = cyclic_group_action(coords, R_perm, num_wedge)
    return cell_complex, action