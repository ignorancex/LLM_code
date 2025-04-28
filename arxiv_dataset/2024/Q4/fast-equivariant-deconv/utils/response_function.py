import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from .spherical_harmonic import _sh_matrix_sh2s

# Everything bellow (CONSTANT_BVALS_DELTA // 2)  is considered b0
# Then B-values are rounded to the nearest multiple of CONSTANT_BVALS_DELTA
# Need to make it more robust, check Mrtrix3 code for bvals clustering
CONSTANT_BVALS_DELTA = 200

class ResponseFunctions():
    def __init__(self, rf_path, rf_anisotropic_names=['wm_response'], rf_isotropic_names=['gm_response', 'csf_response'], normalize_use_tissue='wm_response', normalize_per_shell=False, verbose=False):
        # response functions are assumed to be order by increasing bvals, rows are shells, columns are SHC
        self.verbose = verbose
        self.rf_path = rf_path
        self.normalize_use_tissue = normalize_use_tissue
        self.normalize_per_shell = normalize_per_shell
        if self.verbose:
            print(f'Loading response function: {self.rf_path}')
        assert normalize_use_tissue in (rf_anisotropic_names + rf_isotropic_names) or normalize_use_tissue==''

        # Try to load corresponding bvals. If fails, don't assume anything and later check if number of shells is the same as subject
        self.bvals, self.provided_bvals = self.load_bvals(rf_path)

        # Load anisotropic response functions
        self.rf_anisotropic_names = rf_anisotropic_names
        self.n_anisotropic = len(rf_anisotropic_names)
        if self.n_anisotropic > 0:
            self.rf_anisotropics = self.load_rf_list(rf_anisotropic_names, isotropic=False)
            
        # Load isotropic response functions
        self.rf_isotropic_names = rf_isotropic_names
        self.n_isotropic = len(rf_isotropic_names)
        if self.n_isotropic > 0:
            self.rf_isotropics = self.load_rf_list(rf_isotropic_names, isotropic=True)

        # Get number of shells and SHC
        assert self.n_anisotropic + self.n_isotropic > 0, f'No response function loaded'
        if self.n_anisotropic > 0 and self.n_isotropic > 0:
            assert self.rf_anisotropics.shape[1] == self.rf_isotropics.shape[1], f'Anisotropic and isotropic response functions must have the same number of shells: {self.rf_anisotropics.shape[1]} != {self.rf_isotropics.shape[1]}'
        if self.n_anisotropic > 0:
            self.n_shell = self.rf_anisotropics.shape[1]
            self.n_shc = self.rf_anisotropics.shape[2]
            self.n_degree = 2 * (self.n_shc - 1)
        else:
            self.n_shell = self.rf_isotropics.shape[1]
            self.n_shc = self.rf_isotropics.shape[2]
            self.n_degree = 2 * (self.n_shc - 1)
        if self.verbose:
            print(f'Total shc degree: {self.n_degree}')

        # Update bvals if necessary
        if self.bvals is None:
            self.bvals = np.arange(self.n_shell)
        assert self.n_shell == self.bvals.shape[0], f'Number of shells in response functions ({self.n_shell}) does not match number of shells in bvals ({self.bvals.shape[0]})'
        assert self.bvals[0] == 0, f'First bvals must be 0, got {self.bvals[0]}'

        # Normalization
        self.norm = self.get_normalization()
        if self.n_anisotropic > 0:
            self.rf_anisotropics = self.rf_anisotropics / self.norm
        if self.n_isotropic > 0:
            self.rf_isotropics = self.rf_isotropics / self.norm

    def load_bvals(self, rf_path):
        bvals_path = f'{rf_path}/bvals.bvals'
        if not os.path.exists(bvals_path):
            if self.verbose:
                print(f'bvals not found at: {bvals_path}')
            return None, False
        if self.verbose:
            print(f'Loading bvals: bvals.bvals')
        bvals = np.loadtxt(bvals_path, ndmin=1)
        assert len(bvals.shape) == 1, f'Invalid bvals shape {bvals.shape}'
        bvals = np.rint(bvals / CONSTANT_BVALS_DELTA) * CONSTANT_BVALS_DELTA
        bvals = np.unique(bvals)
        if self.verbose:
            print(f'Loaded {bvals.shape[0]} shells')
        return bvals, True

    def load_rf(self, rf_name, isotropic=False):
        if self.verbose:
            print(f'Loading response function: {rf_name}')
        rf_path = f'{self.rf_path}/{rf_name}.txt'
        assert os.path.exists(rf_path), f'Response function {rf_name} not found at {rf_path}'
        rf = np.loadtxt(rf_path, ndmin=2)
        assert len(rf.shape) == 2, f'Invalid response function {rf_name} shape {rf.shape}'
        if isotropic:
            if rf.shape[1]>1:
                if self.verbose:
                    print(f'Reponse function {rf_name} is not isotropic, only the first column will be used')
                rf = rf[:, 0:1]
        else:
            assert rf.shape[1]>1, f'Reponse function {rf_name} is isotropic, use isotropic=True'
        return rf

    def load_rf_list(self, rf_names, isotropic=True):
        rf_list = []
        for rf_name in rf_names:
            rf = self.load_rf(rf_name, isotropic=isotropic)
            rf_list.append(rf)
        rf_list = np.stack(rf_list)
        n_rf, n_shell, n_shc = rf_list.shape
        if self.verbose:
            print(f'Loaded {n_rf} response functions with {n_shell} shells and {n_shc} SHC')
        return rf_list

    def load_anisotropic_response_functions(self, rf_anisotropic_names):
        if self.n_anisotropic > 0:
            rf_anisotropics = []
            for rf_name in rf_anisotropic_names:
                rf = self.load_rf(rf_name, isotropic=False)
                rf_anisotropics.append(rf)
            rf_anisotropics = np.stack(rf_anisotropics)
            n_anisotropic, n_shell_anisotropic, n_shc_anisotropic = rf_anisotropics.shape
            assert n_anisotropic==self.n_anisotropic, f'Loaded number of anisotropic response function does not match given list: {n_anisotropic} != {self.n_anisotropic}'
            if self.verbose:
                print(f'Loaded {n_anisotropic} anisotropic response functions with {n_shell_anisotropic} shells and {n_shc_anisotropic} SHC')
            return rf_anisotropics

    def get_normalization(self):
        if self.normalize_use_tissue != '':
            if self.normalize_use_tissue in self.rf_anisotropic_names:
                norm = self.rf_anisotropics[self.rf_anisotropic_names.index(self.normalize_use_tissue)]
            else:
                norm = self.rf_isotropics[self.rf_isotropic_names.index(self.normalize_use_tissue)]
            if self.normalize_per_shell:
                norm = norm[:, 0:1]
            else:
                norm = norm[0, 0]
            norm = norm / np.sqrt(4 * np.pi)
        else:
            norm = 1
        if self.verbose:
            print(f'Normalization: {norm}')
        return norm

    def plot_rf(self, isotropic=True):
        # If isotropic is True, plot response function magnitude
        if isotropic:
            self.plot_magnitude()
        else:
            self.plot_sphere()

    def plot_magnitude(self):
        for i in range(self.n_anisotropic):
            plt.plot(self.bvals, self.rf_anisotropics[i, :, 0], label=f'Anisotropic {self.rf_anisotropic_names[i]}')
        for i in range(self.n_isotropic):
            plt.plot(self.bvals, self.rf_isotropics[i, :, 0], label=f'Isotropic {self.rf_isotropic_names[i]}')
        plt.title('Response function magnitude')
        plt.xlabel('B-value')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.show()

    def plot_sphere(self, n_side=4):
        import plotly.graph_objects as go
        npix = hp.nside2npix(n_side)
        indexes = np.arange(npix)
        x, y, z = hp.pix2vec(n_side, indexes, nest=True)
        vector = np.stack([x, y, z], axis=1)
        sh2s = _sh_matrix_sh2s(self.n_degree, vector, with_order=0, symmetric=True)
        if self.n_anisotropic > 0:
            rf_anisotropic_projected = self.rf_anisotropics.dot(sh2s)
        if self.n_isotropic > 0:
            rf_isotropic_projected = self.rf_isotropics.dot(sh2s[0:1])

        fig = go.Figure()
        for j in range(self.n_shell):
            for i in range(self.n_anisotropic):
                fig.add_trace(go.Mesh3d(x=vector[:, 0]+3*j, y=vector[:, 1]+3*0, z=vector[:, 2] + 3*i, alphahull=1, colorscale='jet', intensity=rf_anisotropic_projected[i, j], cmin=0, cmax=np.max([np.max(rf_anisotropic_projected), np.max(rf_isotropic_projected)])))
                fig.update_traces(showlegend=False)
            for i in range(self.n_isotropic):
                fig.add_trace(go.Mesh3d(x=vector[:, 0]+3*j, y=vector[:, 1]+3*0, z=vector[:, 2] + 3*(i+self.n_anisotropic), alphahull=1, colorscale='jet', intensity=rf_isotropic_projected[i, j], cmin=0, cmax=np.max([np.max(rf_anisotropic_projected), np.max(rf_isotropic_projected)])))
                fig.update_traces(showlegend=False)

        layout = go.Layout(
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
                bgcolor='rgba(0, 0, 0, 0)',
                aspectmode='data'
            ),
                width=1000, height=1000,
            scene_camera_eye=dict(x=0, y=-5, z=0)
        )
        fig.update_layout(layout)
        fig.show()


class GroupResponseFunctions():
    def __init__(self, response_function_list, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print('-'*30)
            print('-'*6, f' Loading Group response functions ', '-'*6)
        assert len(response_function_list) > 0, f'Empty response function list'
        self.response_function_list = response_function_list

        # Get number of response functions
        self.n_subject = len(response_function_list)
        if self.verbose:
            print(f'Loaded {self.n_subject} response functions')

        # Check if every subject has the same number of response functions
        self.n_anisotropic = response_function_list[0].n_anisotropic
        self.n_isotropic = response_function_list[0].n_isotropic
        for i in range(self.n_subject):
            assert response_function_list[i].n_anisotropic == self.n_anisotropic, f'Number of anisotropic response functions in response functions ({response_function_list[i].n_anisotropic}) does not match number of anisotropic response functions in first subject ({self.n_anisotropic})'
            assert response_function_list[i].n_isotropic == self.n_isotropic, f'Number of isotropic response functions in response functions ({response_function_list[i].n_isotropic}) does not match number of isotropic response functions in first subject ({self.n_isotropic})'
        if self.verbose:
            print(f'Anisotropic response functions: {self.n_anisotropic}')
            print(f'Isotropic response functions: {self.n_isotropic}')

        # Get maximum number of SHC
        self.max_n_shc = 0
        if self.n_anisotropic > 0:
            for i in range(self.n_subject):
                self.max_n_shc = max(response_function_list[i].n_shc, self.max_n_shc)
            if self.verbose:
                print(f'Maximum number of anisotropic SHC: {self.max_n_shc}')

        # Either all response function has a provided list of bvals and they can be differents
        # or none of them has a provided list of bvals and we assume the bvals are the same for all subjects (this is checked later when loading the subjects)
        self.provided_bvals = response_function_list[0].provided_bvals
        if self.verbose:
            print(f'Provided bvals: {self.provided_bvals}')
        for i in range(self.n_subject):
            assert response_function_list[i].provided_bvals == self.provided_bvals, f'Either all response function has a provided list of bvals and they can be differents or none of them has a provided list of bvals and we assume the bvals are the same for all subjects'
        
        if not self.provided_bvals:
            for i in range(self.n_subject):
                assert response_function_list[i].n_shell == response_function_list[0].n_shell, f'Number of shells in response functions ({response_function_list[i].n_shell}) does not match number of shells in first subject ({response_function_list[0].n_shell})'

        # Get bvals mapping per subject
        self.bvals = np.unique([b for i in range(self.n_subject) for b in response_function_list[i].bvals]).tolist()
        self.n_shell = len(self.bvals)
        bvals_index_per_subject = []
        for i in range(self.n_subject):
            bvals_index_per_subject.append([])
            for j in range(response_function_list[i].n_shell):
                bvals_index_per_subject[i].append(self.bvals.index(response_function_list[i].bvals[j]))
        self.bvals_index_per_subject = bvals_index_per_subject
        if self.verbose:
            print(f'Loaded {self.n_shell} shells: {self.bvals}')

        # Compute mean response functions
        if self.n_anisotropic > 0:
            rf_anisotropic_mean = np.zeros((self.n_subject, self.n_anisotropic, self.n_shell, self.max_n_shc))
            rf_anisotropic_exist = np.zeros((self.n_subject, self.n_anisotropic, self.n_shell, self.max_n_shc))
            for i in range(self.n_subject):
                rf_anisotropic_mean[i][:, self.bvals_index_per_subject[i], 0:response_function_list[i].n_shc] = response_function_list[i].rf_anisotropics
                rf_anisotropic_exist[i][:, self.bvals_index_per_subject[i], 0:response_function_list[i].n_shc] = 1
            self.rf_anisotropic_mean = np.sum(rf_anisotropic_mean, axis=0) / np.sum(rf_anisotropic_exist, axis=0)
            if self.verbose:
                print(f'Anisotropic response functions mean: {self.rf_anisotropic_mean}')
        if self.n_isotropic > 0:
            rf_isotropic_mean = np.zeros((self.n_subject, self.n_isotropic,  self.n_shell, 1))
            rf_isotropic_exist = np.zeros((self.n_subject, self.n_isotropic, self.n_shell, 1))
            for i in range(self.n_subject):
                rf_isotropic_mean[i][:, self.bvals_index_per_subject[i]] = response_function_list[i].rf_isotropics
                rf_isotropic_exist[i][:, self.bvals_index_per_subject[i]] = 1
            self.rf_isotropic_mean = np.sum(rf_isotropic_mean, axis=0) / np.sum(rf_isotropic_exist, axis=0)
            if self.verbose:
                print(f'Isotropic response functions mean: {self.rf_isotropic_mean}')

