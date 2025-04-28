from numba import jit

import numpy as np

import gudhi as gd
from scipy.stats import multivariate_normal

@jit(nopython=True)
def grad_est(n, x_n, y_n, C):
    '''
    Estimates the gradient of the matrix C at point n with respect to x and y

    inputs:

    n 		: point where we are estimating the gradient
    x_n 	: max x-value
    y_n 	: max y-value
    C 		: Matrix under consideration

    outputs:

    x_grad 		: centered difference in x (unless boundary point)
    x_grad_dwn	: first order difference downwind in x
    x_grad_up	: first order difference upwind in x
    y_grad 		: centered difference in y (unless boundary point)
    y_grad_dwn	: first order difference downwind in y
    y_grad_up	: first order difference upwind in y

    '''

    if n[0] == 0:
        # leftmost bdy
        x_grad_dwn = C[(n[0]+1, n[1])]-C[(n[0], n[1])]
        x_grad_up = x_grad_dwn
        x_grad = x_grad_dwn
    elif n[0] == x_n-1:
        # rightmost bdy
        x_grad_dwn = C[(n[0], n[1])]-C[(n[0]-1, n[1])]
        x_grad_up = x_grad_dwn
        x_grad = x_grad_dwn
    else:
        # interior point
        x_grad_dwn = C[(n[0], n[1])]-C[(n[0]-1, n[1])]
        x_grad_up = C[(n[0]+1, n[1])]-C[(n[0], n[1])]
        x_grad = (C[(n[0]+1, n[1])]-C[(n[0]-1, n[1])])/2

    if n[1] == 0:
        # downmost bdy
        y_grad_dwn = (C[(n[0], n[1]+1)]-C[(n[0], n[1])])
        y_grad_up = y_grad_dwn
        y_grad = y_grad_dwn
    elif n[1] == y_n-1:
        # upmost bdy
        y_grad_dwn = (C[(n[0], n[1])]-C[(n[0], n[1]-1)])
        y_grad_up = y_grad_dwn
        y_grad = y_grad_dwn
    else:
        # interior point
        y_grad_dwn = (C[(n[0], n[1])]-C[(n[0], n[1]-1)])
        y_grad_up = (C[(n[0], n[1]+1)]-C[(n[0], n[1])])
        y_grad = (C[(n[0], n[1]+1)]-C[(n[0], n[1]-1)])/2

    return x_grad, x_grad_dwn, x_grad_up, y_grad, y_grad_dwn, y_grad_up


@jit(nopython=True)
def chi_grad_det(n, xn, yn, C_gradx, C_grady):
    '''
    Determines where to sample for up or downwind from the point n based on x- and y- gradients of C

    inputs:

    n 			: point where we are estimating the gradient
    x_n 		: max x-value
    y_n 		: max y-value
    C_gradx 	: Estimate of C gradient in x
    C_grady 	: Estimate of C gradient in y

    outputs:

    n_grad_x_up 	: point to sample for x upwind
    n_grad_x_dwn	: point to sample for x downwind
    n_grad_y_up 	: point to sample for y upwind
    n_grad_y_dwn	: point to sample for y downwind

    '''

    if C_gradx >= 0:
        if n[0] != xn-1:
            n_grad_x_up = (n[0]+1, n[1])
            n_grad_x_dwn = (0, 0)
        else:
            n_grad_x_up = n
            n_grad_x_dwn = (0, 0)
    else:
        if n[0] != 0:
            n_grad_x_up = (0, 0)
            n_grad_x_dwn = (n[0]-1, n[1])
        else:
            n_grad_x_up = (0, 0)
            n_grad_x_dwn = n

    if C_grady >= 0:
        if n[1] != yn-1:
            n_grad_y_up = (n[0], n[1]+1)
            n_grad_y_dwn = (0, 0)
        else:
            n_grad_y_up = n
            n_grad_y_dwn = (0, 0)
    else:
        if n[1] != 0:
            n_grad_y_up = (0, 0)
            n_grad_y_dwn = (n[0], n[1]-1)
        else:
            n_grad_y_up = (0, 0)
            n_grad_y_dwn = n

    return n_grad_x_up, n_grad_x_dwn, n_grad_y_up, n_grad_y_dwn


@jit(nopython=True)
def prob_branch(C):
    '''
    Provides probability of a sprout branching based on the local chemoattractant gradient

    inputs:

    C 	: Chemoattractant density

    outputs:

    prob : probability of branching

    '''

    return 1.0
    '''if C < .1:#0.25:
					return 0
				elif C < .25:#.45:
					return 0.2
				elif C < .4:#0.6:
					return .3
				elif C < .5:#0.68:
					return 0.4
				else:
					return 1.0'''

def param_sweep(N, X, filename=None, iter_num=50, plane_dir='less'):
    '''
    param_sweep

    inputs:
    N : Binary image
    iter_num : number of flooding events to compute

    output
    diag : Birth and death times for all topological features
    '''

    if N.shape != X.shape:
        raise Exception("Shape of N and X must the equal")

    xm, ym = N.shape

    r_range = np.linspace(0, np.max(X), iter_num)

    #if plane_dir == 'greater':
    #    r_range = r_range[::-1]

    st = gd.SimplexTree()

    for k, rr in enumerate(r_range):

        # find nonzero pixels in N that are to the correct direction of the plane
        #if plane_dir == 'less':
        N_update = np.logical_and(N, X <= rr)
        #elif plane_dir == 'greater':
        #    N_update = np.logical_and(N, X >= rr)
        #else:
        #    raise Exception("N_update must be \'less\' or \'greater\'")

        # look for vertical neighbors
        cell_loc = N_update == 1
        a = np.where(cell_loc)
        a = np.hstack((a[0][:, np.newaxis], a[1][:, np.newaxis]))
        locs = a[:, 0] + xm*a[:, 1]
        #locs = xm*a[:,0] + a[:,1]
        for j in locs:
            st.insert([j], filtration=k)

        # look for vertical neighbors
        vert_neighbors = np.logical_and(
            N_update[:-1, :] == 1, N_update[1:, :] == 1)
        a = np.where(vert_neighbors)
        a = np.hstack((a[0][:, np.newaxis], a[1][:, np.newaxis]))
        locs = a[:, 0] + xm*a[:, 1]
        #locs = xm*a[:,0] + a[:,1]
        for j in locs:
            st.insert([j, j+1], filtration=k)

        # look for horizontal neighbors
        horiz_neighbors = np.logical_and(
            N_update[:, :-1] == 1, N_update[:, 1:] == 1)
        a = np.where(horiz_neighbors)
        a = np.hstack((a[0][:, np.newaxis], a[1][:, np.newaxis]))
        locs = a[:, 0] + xm*a[:, 1]

        for j in locs:
            st.insert([j, j+xm], filtration=k)

        # look for diagonal neighbors (top left to bottom right)
        diag_neighbors = np.logical_and(
            N_update[:-1, :-1] == 1, N_update[1:, 1:] == 1)
        a = np.where(diag_neighbors)
        a = np.hstack((a[0][:, np.newaxis], a[1][:, np.newaxis]))
        locs = a[:, 0] + xm*a[:, 1]
        #locs = xm*a[:,0] + a[:,1]
        for j in locs:
            st.insert([j, j+xm+1], filtration=k)

        # look for diagonal neighbors (bottom left to top right)
        diag_neighbors = np.logical_and(
            N_update[1:, :-1] == 1, N_update[:-1, 1:] == 1)
        a = np.where(diag_neighbors)
        a = np.hstack((a[0][:, np.newaxis], a[1][:, np.newaxis]))
        locs = a[:, 0] + xm*a[:, 1]

        for j in locs:
            st.insert([j+1, j+xm], filtration=k)

        st.set_dimension(2)

        # include 2-simplices (looking for four different types of corners)

        for j in np.arange(ym-1):
            for i in np.arange(xm-1):

                # top left corner:
                if N_update[i, j] == 1 and N_update[i+1, j] == 1 and N_update[i, j+1] == 1:
                    st.insert([i + xm*j, (i+1) + xm*j,
                               i + xm*(j+1)], filtration=k)

                # top right corner
                if N_update[i, j] == 1 and N_update[i+1, j] == 1 and N_update[i+1, j+1] == 1:
                    st.insert(
                        [i + j*xm, (i+1)+j*xm, (i+1) + (j+1)*xm], filtration=k)

                # bottom left corner
                if N_update[i, j] == 1 and N_update[i, j+1] == 1 and N_update[i+1, j+1] == 1:
                    st.insert([i + j*xm, i + (j+1)*xm,
                               (i+1) + (j+1)*xm], filtration=k)

                # bottom right corner
                if N_update[i+1, j+1] == 1 and N_update[i+1, j] == 1 and N_update[i, j+1] == 1:
                    st.insert([(i+1) + (j + 1)*xm, (i+1) + j *
                               xm, i + (j + 1)*xm], filtration=k)

    diag = st.persistence()
    st.extend_filtration()
    diagExt = st.extended_persistence()

    if filename is not None:
        data = {}
        data['BD'] = diag
        data['EPD'] = diagExt
        np.save(filename, data)
    return diag

def weight_fun_ramp(x, **options):
    '''
    Weight function for persistence images

    inputs 

    x : function input
    b : max x value

    outputs 

    y: function output
    '''

    b = options.get("b")

    y = np.zeros(x.shape)

    samp = np.where(x <= 0)[0]
    y[samp] = np.zeros(samp.shape)

    samp = np.where(np.logical_and(x > 0, x < b))[0]
    y[samp] = x[samp]/b

    samp = np.where(x >= b)[0]
    y[samp] = np.ones(samp.shape)

    return y


def weight_fun_1(x, **options):
    '''
    Weight function of 1's for persistence images

    inputs 

    x: function input

    outputs 

    y: function output
    '''

    y = np.ones(x.shape)

    return y

class angio_abm:

    def __init__(self,
                 IC='linear',
                 rho=0.34,
                 t_final=4.0,
                 chi=0.38,
                 chemo_rate='const',
                 psi=0.5):

        # parameters
        self.D = .00035
        self.alpha = 0.6
        self.chi = chi
        self.rho = rho
        self.beta = 0.05
        self.gamma = 0.1
        self.eta = 0.1
        self.psi = psi

        self.nb_const = 2.5

        self.chemo_rate = chemo_rate

        # grids

        self.eps1 = 0.45
        self.eps2 = 0.45
        self.k = 0.75
        self.nu = (np.sqrt(5) - 0.1)/(np.sqrt(5)-1)

        self.xn = 201
        self.yn = 201
        self.x = np.linspace(0, 1, self.xn)
        self.y = np.linspace(0, 1, self.yn)
        self.IC = IC
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.dt = 0.01
        self.t_final = t_final
        self.time_grid = np.arange(0, self.t_final, self.dt)

        # initialize sprouts

        self.cell_locs = [[0, .1], [0, .2], [0, .3], [0, .4],
                          [0, .5], [0, .6], [0, .7], [0, .8], [0, .9]]
        self.sprouts = []
        self.sprout_ages = []

        self.branches = 0

        # time-dependent things to save
        self.sprouts_time = []
        self.branches_time = []
        self.active_tips = []

    def IC_generate(self):
        '''
        Sets the initial conditions for n , C, and F. 

        '''

        self.Y, self.X = np.meshgrid(self.y, self.x)

        # TAF
        if self.IC == 'tumor':
            # Equationa 10 & 11 in Anderson-Chaplain
            r = np.sqrt((self.X-1)**2 + (self.Y-0.5)**2)
            self.C = (self.nu - r)**2/(self.nu-0.1) / 1.68
            self.C[r <= 0.1] = 1

        elif self.IC == 'linear':
            # Equation 12 in Anderson-Chaplain

            self.C = np.exp(-((1-self.X)**2)/self.eps1)

        # fibronectin
        # Equation 13 in Anderson-Chaplain
        self.F = self.k*np.exp(-self.X**2/self.eps2)

        # endo_loc
        self.N = np.zeros(self.X.shape)

    def chemotaxis_rate(self, C):
        '''
        Sets the chemotaxis rate (as a function of C) for the model

        outputs:

        chi 	: Chemotaxis rate (const is constant , hill returns hill function).

        '''

        if self.chemo_rate == 'hill':
            return self.chi/(1+self.alpha*self.C[n])
        elif self.chemo_rate == 'const':
            return self.chi

    def sprout_initialize(self):
        '''
        Set the initial sprout locations for the model
        '''

        for c in self.cell_locs:
            self.new_sprout(
                [tuple((np.argmin(np.abs(self.x-c[0])), np.argmin(np.abs(self.y-c[1]))))])

    def record_bio_data(self):
        '''
        Record the number of sprouts, branches, and active tip cells in the model over time

        '''
        self.sprouts_time.append(len(self.sprouts))
        self.branches_time.append(self.branches)
        self.active_tips.append(sum(np.array(self.sprout_ages) != -1))

    def get_bio_data(self):
        '''
        Save the sprouts, branches, tips, time, and overall network to memory

        '''

        data = {}
        data['sprouts'] = self.sprouts_time
        data['branches'] = self.branches_time
        data['active_tips'] = self.active_tips
        data['t'] = self.time_grid
        data['N'] = self.N

        return data

    def move_sprouts(self):
        '''
        Update the tip cell locations and overall network based on tip cell movement.

        '''

        for i, nl in enumerate(self.sprouts):

            # sprout no longer moving if it anastamosed (age = -1 for anasotomosed sprouts)
            if self.sprout_ages[i] == -1:
                continue

            # get current tip cell
            n = nl[-1]

            # sample local gradients
            C_gradx, C_gradx_dwn, C_gradx_up, C_grady, C_grady_dwn, C_grady_up = grad_est(
                n, self.xn, self.yn, self.C)
            F_gradx, F_gradx_dwn, F_gradx_up, F_grady, F_grady_dwn, F_grady_up = grad_est(
                n, self.xn, self.yn, self.F)

            # determine indices of up/down wind based on gradients.
            n_x_up, n_x_dwn, n_y_up, n_y_dwn = chi_grad_det(
                n, self.xn, self.yn, C_gradx_up, C_grady_up)

            # Move tip cells: P0 is the probability a cell stays put, P1-4 are the probabilities
            # of moving right, left, up, and down, respectively.

            # beginning by defining these probabilities from just diffusion
            # start with just diffusion
            P0 = 1.0 - 4.0*self.dt/(self.dx**2)*self.D

            # move right,left,up,down
            P1, P2, P3, P4 = self.dt/(self.dx**2)*self.D, self.dt/(
                self.dx**2)*self.D, self.dt/(self.dx**2)*self.D, self.dt/(self.dx**2)*self.D

            # now incorporate chemotaxis
            # increasing chemical gradient -- sample downwind
            if C_gradx > 0:
                P0 += -self.dt/(self.dx**2) * \
                    (self.chemotaxis_rate(self.C[n])*C_gradx_dwn)
                P1 += self.dt/(self.dx**2) * \
                    (self.chemotaxis_rate(self.C[n_x_up])*C_gradx_up)
            elif C_gradx < 0:
                # deccreasing chemical gradient -- sample upwind
                P0 += self.dt/(self.dx**2) * \
                    (self.chemotaxis_rate(self.C[n])*C_gradx_up)
                P2 += -self.dt / \
                    (self.dx**2) * \
                    (self.chemotaxis_rate(self.C[n_x_dwn])*C_gradx_dwn)
            # Do the same in the y-dimension
            if C_grady > 0:
                P0 += -self.dt/(self.dx**2) * \
                    (self.chemotaxis_rate(self.C[n])*C_grady_dwn)
                P3 += self.dt/(self.dx**2) * \
                    (self.chemotaxis_rate(self.C[n_y_up])*C_grady_up)
            elif C_grady < 0:
                # deccreasing chemical gradient -- sample upwind
                P0 += self.dt/(self.dx**2) * \
                    (self.chemotaxis_rate(self.C[n])*C_grady_up)
                P4 += -self.dt / \
                    (self.dx**2) * \
                    (self.chemotaxis_rate(self.C[n_y_dwn])*C_grady_dwn)

            # haptotaxis
            # increasing chemical gradient, then sample downwind
            if F_gradx > 0:
                P0 += self.rho*self.dt/(self.dx**2)*(F_gradx_dwn)
                P2 += -self.rho*self.dt/(self.dx**2)*(F_gradx_up)
            elif F_gradx < 0:
                # decreasing chemical gradient, then sample upwind
                P0 += -self.rho*self.dt/(self.dx**2)*(F_gradx_up)
                P1 += self.rho*self.dt/(self.dx**2)*(F_gradx_dwn)

            # do the same for y
            if F_grady > 0:
                P0 += self.rho*self.dt/(self.dx**2)*(F_grady_dwn)
                P4 += -self.rho*self.dt/(self.dx**2)*(F_grady_up)

            elif F_grady < 0:
                P0 += -self.rho*self.dt/(self.dx**2)*(F_grady_up)
                P3 += self.rho*self.dt/(self.dx**2)*(F_grady_dwn)

            # now we have our final probabilities
            total = P0 + P1 + P2 + P3 + P4

            # determine random number
            p = np.random.uniform(low=0, high=total)

            if p < P0:
                # stay put
                nl.append(n)
                moved = False
            elif p < P1+P0:
                # move right
                if n[0] <= len(self.x)-2:
                    nl.append((n[0]+1, n[1]))
                    moved = True
                else:
                    nl.append(n)
                    moved = False
            elif p < P2+P1+P0:
                # move left
                if n[0] > 0:
                    nl.append((n[0]-1, n[1]))
                    moved = True
                else:
                    nl.append(n)
                    moved = False
            elif p < P3+P2+P1+P0:
                # move up
                if n[1] < len(self.y)-1:
                    nl.append((n[0], n[1]+1))
                    moved = True
                else:
                    nl.append(n)
                    moved = False
            elif p <= P4+P3+P2+P1+P0:
                # move down
                if n[1] > 0:
                    nl.append((n[0], n[1]-1))
                    moved = True
                else:
                    nl.append(n)
                    moved = False
            else:
                moved = False

            # anastomsis occurs if vessel moves into occupied space
            if self.N[nl[-1]] == 1 and moved == True:
                # list through other sprouts
                for j, sprout in enumerate(self.sprouts):
                    # don't search same sprout
                    if i != j:
                        # cell no longer active
                        if nl[-1] in sprout and self.sprout_ages[j] != -1:
                            self.sprout_ages[i] = -1

            # Update sprout network
            if self.N[nl[-1]] != 1:
                self.N[nl[-1]] = 1

    def update_grids(self):
        '''
        Update F and C using the equations from the appendix of Anderson-Chaplain
        '''

        for i, nl in enumerate(self.sprouts):
            n = nl[-1]

            self.F[n] = self.F[n] * \
                (1-self.dt*self.gamma*1) + self.dt*self.beta*1
            self.C[n] = self.C[n]*(1-self.dt*self.eta*1)

    def branch(self):
        '''
        Determine if a tip cell should branch. If so, determine where daughter cell is placed

        Rules for tip sprouting are outlined in Section 4.1 of Anderson-Chaplain
        '''

        for i, nl in enumerate(self.sprouts):

            # sprout no longer branching if it anastamosed (sprout_ages[i] = -1)
            if self.sprout_ages[i] == -1:
                continue

            # get i-th tip cell.
            n = nl[-1]

            branch = False

            # Rule 1: branch when age over psi
            if self.sprout_ages[i] > self.psi:

                # which neighboring spots are available?
                # sample over the 3x3 grid [x_rang,y_rang] including n.
                if n[0] == 0:
                    x_rang = np.arange(n[0], n[0]+2)
                elif n[0] == self.xn-1:
                    x_rang = np.arange(n[0]-1, n[0]+1)
                else:
                    x_rang = np.arange(n[0]-1, n[0]+2)

                if n[1] == 0:
                    y_rang = np.arange(n[1], n[1]+2)
                elif n[1] == self.yn-1:
                    y_rang = np.arange(n[1]-1, n[1]+1)
                else:
                    y_rang = np.arange(n[1]-1, n[1]+2)

                xx, yy = np.meshgrid(x_rang, y_rang)
                xx = xx.reshape(-1)
                yy = yy.reshape(-1)

                # number of sprout cells in surrounding 3x3 grid
                avail = np.where(self.N[xx, yy] != 0)[0]

                # Rule 2: is there space to branch?
                if len(avail) > 0:

                    # Rule 3: is endothelial density above some threshold?
                    if len(avail/9) > self.nb_const/self.C[n]:

                        # prob_branching based on C

                        pb = prob_branch(self.C[n])

                        # branch with prob pb
                        if np.random.uniform() < pb:

                            # select one of the available spaces for new sprout
                            new_ind = np.random.permutation(len(avail))[0]

                            # new sprout
                            self.new_sprout([(xx[new_ind], yy[new_ind])])
                            branch = True
                            # set age back to zero
                            self.sprout_ages[i] = 0
                            # increase branches
                            self.branches += 1

            # increase age if we never branched
            if branch == False:
                self.sprout_ages[i] += self.dt

    def new_sprout(self, loc):
        '''
        Create new sprout

        inputs :

        loc 	: list of coordinates of starting point of new branch	 
        '''

        self.sprouts.append(loc)
        self.sprout_ages.append(0)
        self.N[loc[0]] = 1

    def plane_sweeping_TDA(self, outfile):
        '''
        compute level-set flooding topology and corresponding persistence image
        '''

        orients = ['left'] 

        for orient in orients:

            if orient == "left":
                plane_dir = 'less'
                indep_var = self.X
            elif orient == "right":
                plane_dir = 'greater'
                indep_var = self.X
            elif orient == "top":
                plane_dir = 'greater'
                indep_var = self.Y
            elif orient == "bottom":
                plane_dir = 'less'
                indep_var = self.Y

            # compute persistence diagram for simulation
            param_sweep(self.N, indep_var, iter_num=51, plane_dir=plane_dir,filename=outfile)
