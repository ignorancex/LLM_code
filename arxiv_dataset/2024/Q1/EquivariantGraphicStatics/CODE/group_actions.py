import math
import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(rho, theta)

def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)

def pol2complex(rho, theta):
    return complex(pol2cart(rho, theta)[0], pol2cart(rho, theta)[1])


#########################################################################################
### Trivial  Group Action ###############################################################
#########################################################################################


class trivial_group_action:
    def __init__(self, coords):
        self.coords = coords
        self.order = 1
        self.rot_order = 1
    
    def presentation_array(self):
        return [0]
    def generator_indices(self):
        return None
    def matrix(self, index):
        return np.identity(2)
    def num_conjugacy_classes(self):
        return 1
    def get_irred_char(self, char_num, index):
        return 1
    def full_char_table(self):
        return np.array([[1]])
    def print_char_table(self):
        print("Trivial Group Action")
        print([1.0])
    def char_inner_product(self, char):
        return char
    def vert_permutation(self, element):
        permuted_vertices = dict()
        for key in range(0, self.coords.shape[0]):
            permuted_vertices[key] = (key, 1)
        return permuted_vertices
    def regular_representation(self, index):
        #The matrix of the left-regular representation of the (index) group element
        return np.array([[1]])
    def regular_char(self):
        #Returns the irreducible character components of the left-regular representation
        #This is equivalently the dimensions of the irreducible characters
        return [1]
    def matrix_char(self):
        return [2]


#########################################################################################
### Cyclic Group Action #################################################################
#########################################################################################


class cyclic_group_action:
    def __init__(self, coords, rot_perm, order):
    #Rotations are COUNTER-CLOCK WISE
        self.coords = coords
        self.rot_perm = rot_perm
        self.order = order
        self.rot_order = order
    
    def presentation_array(self):
        #Gives an lsit of group elements
        #Each group element is presented as a number giving instructions on how functions should use it
        arr = [] * self.order 
        for j in range(0, self.order):
            arr[j] = j
        return arr
    def generator_indices(self):
        return 1
    
    def matrix(self, index):
        #Returns a 2x2 matrix encoding the rotation and or flip in 2D space
        vector = pol2cart(1, 2*math.pi*index/self.order)
        matx = np.zeros((2,2))
        matx[0,0], matx[1,0] = vector[0], vector[1]
        matx[0,1], matx[1,1] = -vector[1], vector[0]
        return matx
    
    def num_conjugacy_classes(self):
        return self.order
    
    def get_irred_char(self, char_num, index):
        #Outputs the character table at the (char_num) row and the (index) column group element
        #These are referenced from a pre-determined standard character table
        root = pol2complex(1, 2*math.pi/self.order)
        return root**(char_num*index)
    
    def full_char_table(self):
        matx = np.zeros((self.num_conjugacy_classes(), self.order), dtype = complex)
        for i in range(0, self.num_conjugacy_classes()):
            for j in range(0, self.order):
                matx[i,j] = self.get_irred_char(i, j)
        return matx
    
    def print_char_table(self):
        char_table = self.full_char_table()
        print("Cyclic group of order:", self.order)
        print("z = order " + str(self.order) + " root of unity")
        for i in range(0, char_table.shape[0]):
            string = ""
            for j in range(0, char_table.shape[1]):
                string += "z^" + str(i*j % self.order) + "  "
            print(string)
    
    def char_inner_product(self, char):
        #Returns the irreducible components of the (char) character
        full_table = self.full_char_table()
        sizer = self.num_conjugacy_classes()
        inner_vals = np.zeros(sizer)
        for i in range(0, sizer):
            summy = 0
            for j in range(0, self.order):
                summy += full_table[i,j] * np.conjugate(char[j])
            if abs(np.round(summy /self.order) - summy / self.order) > 0.00000001:
                #Error in Character inner product
                print("ERROR in cyclic character inner product")
                print("Sum is:", summy, "Group order is:", self.order)
                print("Character:", char)
                print("Character Table:")
                print(full_table)
            inner_vals[i] = int(np.round(np.real(summy) / self.order))
        return inner_vals

    def vert_permutation(self, element):
        #Gives the permutation of vertices under the (element) group element permutation
        #The sign of the transformation is attatched (always 1)
        permuted_vertices = {}
        for key in self.rot_perm:
            #Start with identity permutation
            permuted_vertices[key] = (key, 1)
        
        #Apply rotations, rotate rot_num times
        for i in range(0, element):
            temp = {}
            for key in self.rot_perm:
                to_vert, sign = permuted_vertices[key]
                temp[key] = (self.rot_perm[to_vert], sign)
            permuted_vertices = temp
        
        return permuted_vertices

    def regular_representation(self, index):
        #The matrix of the left-regular representation of the (index) group element
        representation = np.zeros((self.order, self.order))
        for j in range(0, self.order):
            net_rotation = (j + index) % self.rot_order
            i = net_rotation * self.rot_order
            representation[i,j] = 1
        return representation
    
    def regular_char(self):
        #Returns the irreducible character components of the left-regular representation
        #This is equivalently the dimensions of the irreducible characters
        char = np.zeros(self.order)
        char[0] = self.order
        return self.char_inner_product(char)
    
    def matrix_char(self):
        char = np.zeros(self.order)
        for i in range(0, self.order):
            char[i] = np.trace(self.matrix(i))
        return self.char_inner_product(char)


#########################################################################################
### Dihedral Group Action ###############################################################
#########################################################################################


class dihedral_group_action:
    def __init__(self, coords, rot_perm, rot_order, flip_perm):
        #Rotations are COUNTER-CLOCK WISE
        self.coords = coords
        self.rot_perm = rot_perm
        self.flip_perm = flip_perm
        self.rot_order = rot_order #The order of the generating rotation
        self.order = 2 * rot_order #The order of the group
        #Figure out if the flip is a y-axis flip or an x-axis flip
        self.yflip = False
        for i in range(0, self.coords.shape[0]):
            if abs(self.coords[i,1]) > 0.01:
                to_index = self.flip_perm[i]
                to_coord_y = self.coords[to_index, 1]
                if abs(self.coords[i,1] + to_coord_y) < 0.01:
                    self.yflip = True
        self.presentation = self.presentation_array()
                    
    def presentation_array(self):
        #Gives an array of group elements
        #Each group element is presented as a tuple ( , ) giving instructions on how functions should use it
        arr = [()] * self.order 
        for j in range(0, self.order):
            flip_bool = (j >= self.rot_order)
            rot_num = j % self.rot_order
            arr[j] = (rot_num, flip_bool)
        return arr
    
    def generator_indices(self):
        #Returns the generating group element indices
        #Here, returns the index of the single rotation and index of the flip
        return [1, self.rot_order]
    
    def matrix(self, index):
        #Returns a 2x2 matrix encoding the rotation and or flip in 2D space
        rot_num = self.presentation[index][0]
        flip_bool = self.presentation[index][1]
        vector = pol2cart(1, 2*math.pi*rot_num/self.rot_order)
        matx = np.zeros((2,2))
        matx[0,0], matx[1,0] = vector[0], vector[1]
        matx[0,1], matx[1,1] = -vector[1], vector[0]
        flip_matx = np.identity(2)
        if self.yflip:
            flip_matx[1,1] = -1
        else:
            flip_matx[0,0] = -1
        if flip_bool:
            matx = np.dot(flip_matx, matx)
        return matx
    
    def num_conjugacy_classes(self):
        if self.rot_order % 2 == 0:
            return round(self.rot_order/2 + 3)
        else:
            return round((self.rot_order+3)/2)
    
    def get_irred_char(self, char_num, index):
        #Outputs the character table at the (char_num) row and the (index) column group element
        #These are referenced from a pre-determined standard character table
        rot_num = self.presentation[index][0]
        flip_bool = self.presentation[index][1]
        #First give the 1-dim (abelian) characters
        if char_num == 0:
            return 1
        elif char_num == 1:
            if flip_bool:
                return -1
            else:
                return 1
        elif self.rot_order % 2 == 0 and char_num == 2:
            return (-1) ** rot_num
        elif self.rot_order % 2 == 0 and char_num == 3:
            if flip_bool:
                return -( (-1) ** rot_num)
            else:
                return (-1) ** rot_num
        #Done with 1-dim characters, now give 2-dim characters
        if self.rot_order % 2 == 0:
            h = char_num - 4 + 1
        else:
            h = char_num - 2 + 1
        if flip_bool:
            return 0
        else:
            return 2*np.cos(2 * h * rot_num * np.pi / self.rot_order)
    
    def full_char_table(self):
        matx = np.zeros((self.num_conjugacy_classes(), self.order))
        for i in range(0, self.num_conjugacy_classes()):
            for j in range(0, self.order):
                matx[i,j] = self.get_irred_char(i, j)
        return matx
    
    def print_char_table(self):
        char_table = self.full_char_table()
        print("Dihedral Group of order:", self.order)
        for i in range(0, char_table.shape[0]):
            print(char_table[i].round(2))
    
    def char_inner_product(self, char):
        #Returns the irreducible components of the (char) character
        full_table = self.full_char_table()
        sizer = self.num_conjugacy_classes()
        inner_vals = np.zeros(sizer)
        for i in range(0, sizer):
            summy = 0
            for j in range(0, self.order):
                #Technically, should be complex conjugate multiplication here
                summy += full_table[i,j]*char[j]
            if abs(np.round(summy /self.order) - summy / self.order) > 0.00000001:
                #Error in Character inner product
                print("ERROR in dihedral character inner product")
                print("Sum is:", summy, "Group order is:", self.order)
                print("Character:", char)
                print("Character Table:")
                print(full_table)
            inner_vals[i] = int(np.round(np.real(summy) / self.order))
            inner_vals[i] = int(np.round(summy / self.order))
        return inner_vals

    def vert_permutation(self, element):
        #Gives the permutation of vertices under the (elment) group element permutation
        #The sign of the transformation is attatched (always 1)
        rot_num = self.presentation[element][0]
        flip_bool = self.presentation[element][1]
        permuted_vertices = {}
        for key in self.rot_perm:
            #Start with identity permutation
            permuted_vertices[key] = (key, 1)
        
        #Apply rotations, rotate rot_num times
        for i in range(0, rot_num):
            temp = {}
            for key in self.rot_perm:
                to_vert, sign = permuted_vertices[key]
                temp[key] = (self.rot_perm[to_vert], sign)
            permuted_vertices = temp
        
        #Apply flip
        if flip_bool:
            temp = {}
            for key in self.flip_perm:
                to_vert, sign = permuted_vertices[key]
                temp[key] = (self.flip_perm[to_vert], sign)
            permuted_vertices = temp
        
        return permuted_vertices
    
    def regular_representation(self, index):
        #The matrix of the left-regular representation of the (index) group element
        rot_num = self.presentation[index][0]
        flip_bool = self.presentation[index][1]
        representation = np.zeros((self.order, self.order))
        for j in range(0, self.order):
            flipped = (j >= self.rot_order)
            rotated = j % self.rot_order
            if flipped: #Rotations commute past the flipping
                net_rotation = (rotated -rot_num) % self.rot_order
            else:
                net_rotation = (rotated + rot_num) % self.rot_order
            net_flip = (flipped != flip_bool)
            i = net_rotation + net_flip * self.rot_order
            representation[i,j] = 1
        return representation
    
    def regular_char(self):
        #Returns the irreducible character components of the left-regular representation
        #This is equivalently the dimensions of the irreducible characters
        char = np.zeros(self.order)
        char[0] = self.order
        return self.char_inner_product(char)
    
    def matrix_char(self):
        char = np.zeros(self.order)
        for i in range(0, self.order):
            char[i] = np.trace(self.matrix(i))
        return self.char_inner_product(char)