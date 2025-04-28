#!/usr/bin/env python3

# A single-qudit class - to be used for ECT tomography
# Mostly used to provide observables (the SU(d) generators) and the relative projectors.
#
# Generator 0 is the identity and represents all of the maximum abelian subspace
# Generators from 1          to d(d-1)/2 INCLUDED are the real ones
# Generators from d(d-1)/2+1 to d(d-1)   INCLUDED are the imaginary ones

# (c) 2024 Giovanni Garberoglio (garberoglio@ectstar.eu), Daniele Binosi (binosi@ectstar.eu)

import numpy as np

class singleQudit():

    def __init__(self, d):
        print("Setting up a qudit with d=",d)
        self.d = d
        self.triu_indices = np.triu_indices(d,k=1)
        mel_re = [(i,j,'r') for i,j in zip(*self.triu_indices)]
        mel_im = [(i,j,'i') for i,j in zip(*self.triu_indices)]
        self.mel = mel_re + mel_im
        self.imaginary_generator_offset = int(d*(d-1)/2)
        
        self.idx_im = list(range(1+len(mel_re),1+len(mel_re)+len(mel_im)))
        
        print("Got %d single-qudit real observables" % (int(d*(d-1)/2)))
        print("and %d elements of the maximum abelian subgroup" % (d-1))
        self.nsettings = len(self.mel)+1

        self.operator_idx = [(0,0)] + [(i,j) for i,j in zip(*self.triu_indices)]
        
        self.generator_names = ['Z']
        self.generator_names = self.generator_names + [f'R%d.%d' % (i,j) for i,j in zip(*self.triu_indices)]
        self.generator_names = self.generator_names + [f'I%d.%d' % (i,j) for i,j in zip(*self.triu_indices)]

    def get_generator(self, n):
        """
        get the nth generator.
        The first (n=0) is the identity (corresponding to the diagonal)        
        Then we have the d(d-1)/2 real generator
        Then we have the d(d-1)/2 imaginary ones        
        """
        d = self.d
        if n == 0: # setting corresponding to the computational basis
            return(np.eye(self.d))
        else:
            i,j,im_or_re = self.mel[n-1]
            O = np.zeros((d,d), dtype=np.complex128)
            if im_or_re == 'r':
                a = 1
                b = 1
            else:
                a = 1j
                b = -1j                
            O[i,j] = a
            O[j,i] = b
            return(O)

    def get_generator_name(self, n):
        return self.generator_names[n]
            
    def get_generator_number_from_indices(self, i_d, j_d, re_or_im='r'):
        if i_d > j_d:
            i_d, j_d = j_d, i_d
            
        if re_or_im == 'r':
            offset = 0
        else:
            offset = self.imaginary_generator_offset
            
        if i_d == j_d:
            ret = 0
        else:
            ret = offset + self.operator_idx.index((i_d,j_d))

        return(ret)
            
    def get_projectors_names(self, n):
        """
        the projectors are named in the following way:
        [letter][i].[j] where i,j are numbers

        We start with a d-dimensional vector v[] of all zeros and fill it according to the following
        convention (where isq=1/√2)
        letter = z: (i must be = j) -> v[i] = 1.0 (element of the computational basis)
        letter = X: v[i] = isq, v[j] =     isq (+1 eigenvector of the real observable (i,j))
        letter = x: v[i] = isq, v[j] =    -isq (-1 eigenvector of the real observable (i,j))
        letter = Y: v[i] = isq, v[j] =  1j*isq (+1 eigenvector of the imag observable (i,j))
        letter = y: v[i] = isq, v[j] = -1j*isq (-1 eigenvector of the imag observable (i,j))                      
        """
        if n==0:
            ret=['Z'+str(i)+'.'+str(i) for i in range(self.d) ]
        else:
            ret = []            
            i,j,im_or_re = self.mel[n-1]
            if im_or_re == 'r':
                symbols = ['X','x']
            else:
                symbols = ['Y','y']

            for n in range(self.d):
                name='Z'+str(n)+'.'+str(n)
                if n==i:
                    name=symbols[0]+str(i)+'.'+str(j)
                if n==j:
                    name=symbols[1]+str(i)+'.'+str(j)

                ret.append(name)

        return(ret)
            
    def projector_from_name(self, name):
        """
        Here we build the projector corresponding to a given name
        """
        isq = 1.0/np.sqrt(2.0)
        proj_vals = {'Z': 1.0,
                     'X': [isq, isq],
                     'x': [isq, -isq],
                     'Y': [isq, 1j*isq],
                     'y': [isq, -1j*isq]}
        
        proj_type = name[0]
        proj_idx = list(map(int,name[1:].split('.')))
        
        ret = np.zeros(self.d, dtype=np.complex128)
        
        ret[proj_idx] = proj_vals[proj_type]
        return(ret);

    def get_projectors(self, n):
        names = self.get_projectors_names(n)
        return(np.array([self.projector_from_name(x) for x in names]))
    
