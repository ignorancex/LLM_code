#!/usr/bin/env python3

# Enhanced Compressive Threshold Quantum State Tomography for qudits
# ECT-Tomography + qudits
# (c) 2024 Giovanni Garberoglio (garberoglio@ectstar.eu), Daniele Binosi (binosi@ectstar.eu)

# This is a class that computes the settings for ECT tomography and optionally returns the projectors

import numpy as np
import itertools as it
import functools as ft
import prettytable as pt

import single_qudit as sqd

def to_base_d(number, base, N=0):
    """
    Convert a number to a list of its digits in base d with N digits.
    N=0 means as many as needed
    """
    if number == 0:
        return [0] * N
    digits = []
    while number:
        digits.append(number % base)
        number //= base
    while len(digits) < N:
        digits.append(0)
    return digits[::-1]

def find_rows_greedy_match(A, random=False, verbose=False):
    """
    Given a matrix A of POSITIVE NUMBERS (possibly of 0s and 1s)
    finds a set of rows such that every element of their sum is at least as large as the largest
    element in the corresponding column using a greedy algorithm.
    
    We progressively consider those rows with the MINIMUM number of zeros
    until the sum of the rows does not have any element equal to zero.
    If random is True, then equivalent rows are chosen randomly.
    Otherwise, the first row that maximizes the number of 1s is taken.
    """

    # check that the sum of the rows of A does not have any zero 
    row_sum = np.sum(A,axis=0)
    zeros_of_sum = np.sum(np.isclose(row_sum,0.0))
    if  zeros_of_sum > 0:
        print("find rows greedy: no solution possible")
        print("Have %d remaining zeros out of %d" % (zeros_of_sum, A.shape[1]))
        exit()

    target = np.max(A,axis=0) # the max on each column

    MAX_ZEROS = A.shape[1]
    rows_considered = []
    columns_to_consider = np.ones(A.shape[1],dtype=bool)
    
    while True:

        # number of zeros for each row for the columns to consider
        nzeros = np.sum( np.isclose(A[:,columns_to_consider], 0) , axis=1)
        nzeros[rows_considered] = MAX_ZEROS # so that the rows already considered are not considered

        x = np.min(nzeros)
        idxs, *_ = np.where(np.isclose(nzeros, x))

        if random == True:
            idx = np.random.choice(idxs)
        else:
            idx = idxs[0]

        rows_considered.append(idx)

        # check if the sum of rows has any zeros
        sum_of_rows = np.sum(A[rows_considered],axis=0)
        this_row = A[idx]
        columns_to_remove , *_ = np.where(sum_of_rows >= target)
        columns_to_consider[columns_to_remove] = False

        if np.all(sum_of_rows >= target):
            break
            
    if verbose == True:
        print("found %d settings" % (len(rows_considered)))
        
    mask = np.zeros(A.shape[0],dtype=bool)
    mask[rows_considered] = True
    return(mask, rows_considered)

class ECTtomography():

        def __init__(self, d, N, verbose=False):
            print("Setting up ECT tomography of %d qudits (d=%d)" % (N,d))
            self.sq = sqd.singleQudit(d)
            self.d = d
            self.N = N
            self.size = d**N
            self.verbose = verbose

            self.settings   = []
            self.projs      = []
            self.counts     = []

        def setting_name(self, setting):
            setting_name = [self.sq.get_generator_name(x) for x in setting]
            return(setting_name)
        
        def projectors_of_setting(self, setting):
            """
            For a given setting s returns the d^N projectors | phi^(s)_k >
            """
            phi_single = [ self.sq.get_projectors(s) for s in setting ]
            phi_setting = ft.reduce(np.kron,phi_single)
            return(phi_setting)

        def projector_names_of_setting(self, setting):
            names_single = [ self.sq.get_projectors_names(s) for s in setting ]
            names = list(it.product(*names_single))
            return(names)

        def print_projector_names_of_setting(self, setting):
            pr_names = self.projector_names_of_setting(setting)
            setting_name = self.setting_name(setting)
            setting_name = '(' + ' '.join(setting_name) + ')'
            
            print('Projectors for setting', setting_name, 'of d=', self.d, 'qudits')
            if self.d == 2:
                table = pt.PrettyTable(["Projector","Standard form"])
                D = {'Z0.0': 'H', 'Z1.1': 'V', 'X0.1': 'D', 'x0.1': 'A', 'Y0.1': 'R', 'y0.1': 'L'}
                for p_n in  pr_names:
                    old_names = [D[x] for x in p_n]
                    table.add_row([' '.join(p_n), '|' + ''.join(old_names) + '>'])
            else:
                table = pt.PrettyTable(["Projector"])
                for p_n in  pr_names:
                    table.add_row([' '.join(p_n)])

            print(table)
            print('Name conventions are as follows:')
            print(' * |Zn.n> are vectors of zeroes, except at position n where they are 1;')
            print(' * |Xn.m> and |xn.m> are the eigenvectors with eigenvalues 1 and -1, respectively, of the SU(d) matrices that have 1 in position nm and mn;')
            print(' * |Yn.m> and |yn.m> are the eigenvectors with eigenvalues 1 and -1, respectively, of the SU(d) matrices that have i in position nm and -i at mn.')
            

        def projection_operators_of_setting(self, setting):
            """
            For a given setting s returns P^(s,k)_ij as the array
            P[k,i,j]
            This uses a lot of memory!
            """
            phi_setting = self.projectors_of_setting(setting)
            print("phi of setting",setting,"=",phi_setting.shape)
            P = phi_setting[:,:,None] * np.conj(phi_setting[:,None,:])
            return(P)

        def projector_from_name(self, name):
            single_qudit_projs = [self.sq.projector_from_name(x) for x in name]
            proj = ft.reduce(np.kron,single_qudit_projs)
            return(proj)
        
        def re_setting_to_im(self, s):
            offset = self.sq.imaginary_generator_offset
            s_im = [ x for x in s ]
            for i in range(len(s_im)):
                if s_im[i] > 0:
                    s_im[i] += offset
                    break
            return(tuple(s_im))

        def off_diagonal_matrix_elements_to_be_measured(self, diagonal, threshold):

            d = diagonal#/np.sum(diagonal)
            rho_exp = np.sqrt(np.outer(d,d))
            above_t = (rho_exp >= threshold)
            above_t[np.tril_indices(d.size)] = False
            self.off_diagonal_indices_above_threshold = np.where(above_t)

            idx = self.off_diagonal_indices_above_threshold
            mel_re = [(i,j,'r') for i,j in zip(*idx) if j>i]
            mel_im = [(i,j,'i') for i,j,_ in mel_re]
            mel = mel_re + mel_im
            return(mel)

        def setting_of_mel(self, i, j, re_or_im='r'):
                i_d = to_base_d(i, self.d, self.N)
                j_d = to_base_d(j, self.d, self.N)
                setting = tuple( [self.sq.get_generator_number_from_indices(x,y) for x,y in zip(i_d,j_d)] )
                if re_or_im == 'i':
                    setting = self.re_setting_to_im(setting)
                return(setting)
            
        def all_settings_to_be_measured(self, diagonal, threshold):            
            mel = self.off_diagonal_matrix_elements_to_be_measured(diagonal, threshold)
            mel_re = [ m for m in mel if m[2] == 'r' ]
            
            settings_re = []
            weights_re  = []
            for m in mel_re:
                setting = self.setting_of_mel(*m)
                settings_re.append( setting )
                i,j,_ = m
                weights_re.append( np.sqrt(diagonal[i]*diagonal[j]) )

            settings_im = [ self.re_setting_to_im(s) for s in settings_re ]
            weights_im  = weights_re
            
            all_settings = settings_re + settings_im
            all_weights  = weights_re + weights_im
            return(all_settings, all_weights)

        def unique_settings_to_be_measured(self, diagonal, threshold):
            
            all_settings, all_weights = self.all_settings_to_be_measured(diagonal, threshold)
            
            unique_settings = tuple(set(all_settings))
            unique_weights = [max(all_weights[i] for i, s in enumerate(all_settings) if s == x) for x in unique_settings]

            if self.verbose == True:
                print("Found %d unique settings" % (len(unique_settings)))
            # sort according to the weight
            idx = np.flip(np.argsort(unique_weights))
            
            sorted_settings = [ unique_settings[x] for x in idx ]
            sorted_weights  = [ unique_weights[x] for x in idx]        
            return(sorted_settings, sorted_weights)

        def ECT_settings(self, diagonal, threshold):
            """
            Returns the settings to be used in ECT tomography
            """
            S, W = self.unique_settings_to_be_measured(diagonal, threshold)
            print("starting pruning")
            # pruning
            A = []        
            for s in S:
                idx = self.off_diagonal_indices_above_threshold

                #
                # Reference implementation that uses *A LOT* of memory
                #
                #P = self.projection_operators_of_setting(s)
                #P = P[:,idx[0],idx[1]]
                #Pre = np.sum(np.real(P)**2, axis=0).flatten()
                #Pim = np.sum(np.imag(P)**2, axis=0).flatten()

                p = self.projectors_of_setting(s)
                P = p[:,idx[0]] * np.conj(p[:,idx[1]])
                Pre = np.sum(np.real(P)**2, axis=0).flatten()
                Pim = np.sum(np.imag(P)**2, axis=0).flatten()

                A.append( np.hstack([Pre, Pim]) )    

            A = np.array(A)
            if self.verbose == True:
                print("Pruning matrix A...")
                for s,a in zip(S,A):
                    print(s,a)
                
                print(A)
                
            mask, idx = find_rows_greedy_match(A,verbose=self.verbose)

            pruned_S = [s for s,m in zip(S,mask) if m == True ]
            if self.verbose == True:
                print("Pruned settings")
                for s in pruned_S:
                    print(s)
                
            pruned_W = [w for w,m in zip(W,mask) if m == True ]

            scomb = sorted(list(zip(pruned_W, pruned_S)), key=lambda x: (-x[0], sum(x[1])))
            pruned_S_real_first = [setting for _, setting in scomb]

            if self.verbose == True:
                print("Settings after pruning: ",len(pruned_S))
                print("Settings sequence to be measured:")
                table = pt.PrettyTable(["Weight", "Setting"])
                for pr_w, pr_s in scomb:                    
                    setting_name = self.setting_name(pr_s)
                    setting_name = '(' + ' '.join(setting_name) + ')'                    
                    table.add_row([format(pr_w.real, '.6f'), setting_name])
                print(table)
                print('where:')
                print(' * Weight is the value of \sqrt(rho_ii rho_jj).') 
                print(' * Z is the computational basis (any element of the maximum abelian subgroup);')
                print(' * Rm.n are the real observables (0 everywhere but for a 1 at position mn and nm);')
                print(' * Im.n are the imaginary observables (0 everywhere but for an i at position mn and -i at nm)')

            self.settings = pruned_S_real_first
            self.projs    = [ v for v in self.projectors_of_setting([0]*self.N) ] #add projectors of the diagonal 
            self.counts   = [ np.real(x) for x in diagonal ] #add counts of the diagonal

            return(pruned_S_real_first, pruned_W)

        def add_count(self, proj_name, count):
            proj = self.projector_from_name(proj_name)
            self.projs.append(proj)
            self.counts.append(count)


        def projector_of_matrix_element(self, i, j, re_or_im='r'):
            s = self.setting_of_mel(i,j,re_or_im)
            phi = self.projectors_of_setting(s)
            names = self.projector_names_of_setting(s)
            
            # A_n = (Re or Im)(<i|phi_n^{(s)}> <phi_n^{(s)}|j>)
            if re_or_im == 'r':
                A = np.real(phi[:,i] * np.conj(phi[:,j]))**2
            else:
                A = np.imag(phi[:,i] * np.conj(phi[:,j]))**2                
            m = np.max(A)
            indices = np.where(A == m)[0] # indices of maximum
            idx = indices[0] # get the first

            assert np.allclose(phi[idx],
                               ft.reduce(np.kron, [self.sq.projector_from_name(x) for x in names[idx]]))
            
            return( phi[idx], names[idx])
        
        def tQST_projectors(self, diagonal, threshold):
            """
            Returns the tQST projectors to be used
            - as the rows of an array
            - as a list of names
            """
            mel = self.off_diagonal_matrix_elements_to_be_measured(diagonal, threshold)
            
            off_diagonal_projs = []
            off_diagonal_projs_names = []
                
            for m in mel:
                proj, name = self.projector_of_matrix_element(*m)
                off_diagonal_projs.append(proj)
                off_diagonal_projs_names.append(name)
            
            if self.verbose == True:
                print("Off-diagonal projectors to be measured given the provided diagonal and threshold:")
                if self.d == 2:
                    table = pt.PrettyTable(["Element", "Projector","Standard form"])
                    D = {'Z0.0': 'H', 'Z1.1': 'V', 'X0.1': 'D', 'x0.1': 'A', 'Y0.1': 'R', 'y0.1': 'L'}
                    for m_n, p_n in zip(mel, off_diagonal_projs_names):
                        name_list = p_n #.split(' ')
                        old_names = [D[x] for x in name_list]
                        table.add_row([m_n, p_n, '|' + ''.join(old_names) + '>'])
                else:
                    table = pt.PrettyTable(["Element", "Projector"])
                    for m_n, p_n in zip(mel, off_diagonal_projs_names):
                        table.add_row([m_n, p_n])
                
                print(table)

                print('Name conventions are as follows:')
                print(' * |zn.n> are vectors of zeroes, except at position i where they are 1;')
                print(' * |xn.m> and |Xn.m> are the eigenvectors with eigenvalues -1 and 1, respectively, of the SU(d) matrices that have 1 in position nm and mn;')
                print(' * |yn.m> and |Yn.m> are the eigenvectors with eigenvalues -1 and 1, respectively, of the SU(d) matrices that have i in position nm and -i at mn.')
            
            self.projs    = [ v for v in self.projectors_of_setting([0]*self.N) ] #add projectors of the diagonal 
            self.counts   = [ np.real(x) for x in diagonal ] #add counts of the diagonal

            return(np.array(off_diagonal_projs), off_diagonal_projs_names)
        
        def get_counts(self):
            return np.array(self.projs), np.array(self.counts)

if __name__ == '__main__':
    
    import density_matrix_tool as dmt

    d=4
    N=4
    ect = ECTtomography(d,N,verbose=True)
    diagonal = np.zeros(d**N)
    diagonal[0] = 1/2
    diagonal[5] = 1/3
    diagonal[2] = 1/12
    diagonal[3] = 1/12 # [4] prune from 12 to 10 | [3] 6 independent
    threshold = 1/14
    #projs, names = ect.tQST_projectors(diagonal, threshold)
    #print(projs)
    #print(names)

    S, W = ect.all_settings_to_be_measured(diagonal, threshold)
    print("ALL SETTINGS",len(S));
    print(S)
    for s,w in zip(S,W):
        print("setting",s,"with weight",w)
    
    S, W = ect.unique_settings_to_be_measured(diagonal, threshold)
    print("UNIQUE SETTINGS",len(S));
    for s,w in zip(S,W):
        print("setting",s,"with weight",w)

    print("Finding ECT settings")
    S, W = ect.ECT_settings(diagonal,threshold)
    print("PRUNED",len(S))
    for s,w in zip(S,W):
        print("setting",s,"with weight",w)
    

    exit(3)
    
    d = 3
    N = 3
    ndiag = np.random.randint(3,int(d**N/2))
    print("Random state of d=%d N=%d with %d diagonal elements" % (d,N,ndiag))
    diagonal = np.abs(dmt.random_pure_state(d**N, ndiag=ndiag))**2
    threshold = 0.9 * np.min(diagonal[diagonal>0.0])
    print("diagonal",diagonal)
    print("threshold",threshold)

    ect = ECTtomography(d,N, verbose=True)


    S, W = ect.ECT_settings(diagonal,threshold)
    for s,w in zip(S,W):
        print("setting",s,"with weight",w)

