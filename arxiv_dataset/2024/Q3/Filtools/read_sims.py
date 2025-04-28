import numpy as np
import h5py
import pandas as pd
import sys

class gadget():
    def __init__(self, file_array, l=300):
        self.file_array=file_array
        self.l=l

    def W(self,a,b,c, nbox):
        ''' gives the weight of the box given its x y and z indices and the number of cells along an axis''' 
        a=a%nbox #periodic boundary conditions
        b=b%nbox
        c=c%nbox
        return (a*nbox**2+b*nbox+c)

##############################################################################################


    def get_pos(self):
        '''This function gives the positions of DM particles in the given gadget files
        file_array must be a list of the files containing the gadget output'''

        #loading the data
        prtcl_types = ["Gas","Halo","Disk",  "Bulge", "Stars", "Bndry"]
        prtcl_type="Halo"
        file = open(self.file_array[0],'rb')

        header_size = np.fromfile(file, dtype=np.uint32, count=1)
        N_prtcl_thisfile = np.fromfile(file, dtype=np.uint32, count=6)    ## The number of particles of each type present in the file

        file.seek(256+8, 0)
        position_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]

        N_prtcl = N_prtcl_thisfile[1] #halo particles
        i = 0
        while prtcl_types[i] != prtcl_type:
            file.seek(N_prtcl_thisfile[i]*3*4, 1)
            i += 1
        posd = np.fromfile(file, dtype = np.float32, count = N_prtcl*3)
        posd = posd.reshape((N_prtcl, 3))  
        Pos=pd.DataFrame(posd)
        file.close()

        #extending the data
        for j in range(1,len(self.file_array),1):
                file = open(self.file_array[j],'rb')

                header_size = np.fromfile(file, dtype=np.uint32, count=1)
                N_prtcl_thisfile = np.fromfile(file, dtype=np.uint32, count=6) # The number of particles of each type present in the file

                file.seek(256+8, 0)
                position_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]


                N_prtcl = N_prtcl_thisfile[1]
                i = 0
                while prtcl_types[i] != prtcl_type:
                    file.seek(N_prtcl_thisfile[i]*3*4, 1)
                    i += 1

                posd = np.fromfile(file, dtype = np.float32, count = N_prtcl*3)
                posd = posd.reshape((N_prtcl, 3))
                df_x=pd.DataFrame(posd)
                Pos=pd.DataFrame(np.concatenate([Pos, df_x]))
                file.close()

        Pos=np.array(Pos)
        return Pos

    def get_vel(self):
        '''This function gives the velocities of DM particles in the given gadget files
        g must be a list of the files containing the gadget output'''

        #loading the velocity data
        prtcl_types = ["Gas","Halo","Disk",  "Bulge", "Stars", "Bndry"]
        prtcl_type="Halo"
        Vel1=[]
        for k in range(len(self.file_array)):
            file = open(self.file_array[k],'rb')

            header_size = np.fromfile(file, dtype=np.uint32, count=1)
            N_prtcl_thisfile = np.fromfile(file, dtype=np.uint32, count=6)    ## The number of particles of each type present in the file


            file.seek(256+8+8 + int(N_prtcl_thisfile.sum())*3*4, 0)
            velocity_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]
            i = 0
            while prtcl_types[i] != prtcl_type:
                file.seek(N_prtcl_thisfile[i]*3*4, 1)
                i += 1
            N_prtcl = N_prtcl_thisfile[i]
            veld = np.fromfile(file, dtype = np.float32, count = N_prtcl*3) 
            veld = veld.reshape((N_prtcl, 3))

            Vel1.append(veld)
            file.close()
        Vel=np.concatenate([Vel1[0], Vel1[1], Vel1[2], Vel1[3], Vel1[4], Vel1[5], Vel1[6], Vel1[7]])
        return Vel


    def split(self,Pos=None, nbox=30):
        """ This function gives the indices of the particles such that they are arranged into small sub-boxes"""
        lbox=self.l/nbox
        edge=np.array(np.linspace(lbox, self.l, nbox)) #left edge of the sub-boxes
        if Pos is None:
            Pos=self.get_pos()
        rind=np.array(Pos/lbox, dtype=int) #(i,j,k) index of the box that the particle belongs to
        weight=rind[:,0]*nbox**2+rind[:,1]*nbox+rind[:,2] # index of the box, when the 3D array is flattened
        rind=0  #to save space

        sort=weight.argsort()  #finally, one has to do Pos[sort], Vel[sort]
        weight=weight[sort]
        Num_W=np.bincount(weight)  #number of particles in each sub-box
        weight=0
        Num_cum=np.cumsum(Num_W) #number of particles below the given sub-box 
        #make sure that there are non-zero particles per sub-box, otherwise Num-cum has dimensions different from number of sub-boxes, and the code will fial.
        Num_cum=np.roll(Num_cum, 1)
        Num_cum[0]=0
        
        return sort, Num_cum, Num_W


#####################################################################################################################
