import os
import numpy as np

# Everything bellow (CONSTANT_BVALS_DELTA // 2)  is considered b0
# Then B-values are rounded to the nearest multiple of CONSTANT_BVALS_DELTA
# Need to make it more robust, check Mrtrix3 code for bvals clustering
CONSTANT_BVALS_DELTA = 200

class Gradient():
    def __init__(self, gradient_path, bvecs_name='bvecs.bvecs', bvals_name='bvals.bvals', mask_name=None, Rinv=None, verbose=False):
        # Print option
        self.verbose = verbose
        # Gradient path
        self.gradient_path = gradient_path
        # B-Gradients and B-Values names
        self.bvecs_name = bvecs_name
        self.bvals_name = bvals_name
        # Mask name for sub-sampling
        self.mask_name = mask_name

        # Load B-Gradients and B-Values
        self.bvecs = self.load_bvecs()
        self.bvals, self.unique_bvals = self.load_bvals()

        # If applicable, load B-Mask
        self.mask, self.bvals_masked, self.bvecs_masked, self.unique_bvals_masked = self.load_gradient_mask()

        # Apply affine transform
        self.Rinv = self.apply_affine(Rinv)

    def load_bvecs(self):
        bvecs_path = f'{self.gradient_path}/{self.bvecs_name}'
        if self.verbose:
            print(f'Loading B-Gradients at {bvecs_path}')
        assert os.path.exists(bvecs_path), f'B-Gradients not found at {bvecs_path}'
        bvecs = np.loadtxt(bvecs_path)
        if bvecs.shape[1] != 3:
            assert bvecs.shape[0] == 3, f'Invalid B-Gradients shape {bvecs.shape}'
            bvecs = bvecs.T
        bvecs = bvecs / (np.linalg.norm(bvecs, axis=1, keepdims=True) + 1e-16)
        if self.verbose:
            print(f'Loaded {bvecs.shape[0]} B-Gradients')
        return bvecs

    def load_bvals(self):
        bvals_path = f'{self.gradient_path}/{self.bvals_name}'
        if self.verbose:
            print(f'Loading B-Values at {bvals_path}')
        assert os.path.exists(bvals_path), f'B-Values not found at {bvals_path}'
        bvals = np.loadtxt(bvals_path)
        assert len(bvals.shape) == 1, f'Invalid B-Values shape {bvals.shape}'
        assert bvals.shape[0] == self.bvecs.shape[0], f'Invalid B-Values shape {bvals.shape} and B-Gradients shape {self.bvecs.shape}'
        bvals = np.rint(bvals / CONSTANT_BVALS_DELTA) * CONSTANT_BVALS_DELTA
        assert np.min(bvals) == 0, f'Invalid B-Values {np.min(bvals)} - Minimum B-Value must be 0'
        unique_bvals = np.unique(bvals)
        if self.verbose:
            print(f'Loaded {bvals.shape[0]} B-Values - {unique_bvals.shape[0]} shells: {unique_bvals}')
        # Set bvecs to 0 if bvals == 0
        self.bvecs[bvals==0] = 0
        return bvals, unique_bvals

    def apply_affine(self, Rinv=None):
        if not Rinv is None:
            if self.verbose:
                print(f'Rotating B-Gradients: {Rinv}')
            assert np.abs((np.abs(np.linalg.det(Rinv)) - 1))<1e-2, f'Invalid affine transform {Rinv} - Determinant is {np.linalg.det(Rinv)} but must be 1'
            self.bvecs = np.dot(Rinv, self.bvecs.T).T
            self.bvecs_masked = np.dot(Rinv, self.bvecs_masked.T).T
        else:
            if self.verbose:
                print(f'No affine transform applied')
            Rinv = np.eye(3)
        return Rinv

    def load_gradient_mask(self):
        mask_path = f'{self.gradient_path}/{self.mask_name}'
        if (not self.mask_name is None) and os.path.exists(mask_path):
            if self.verbose:
                print(f'Apply B-Mask at {mask_path}')
            mask = np.loadtxt(mask_path)
            assert len(mask.shape) == 1, f'Invalid B-Mask mask shape {mask.shape}'
            assert mask.shape[0] == self.bvals.shape[0], f'Invalid B-Mask shape {mask.shape} and B-Values shape {self.bvals.shape}'
            bvals_masked = self.bvals[mask>0]
            bvecs_masked = self.bvecs[mask>0]
            unique_bvals_masked = np.unique(bvals_masked)
            if self.verbose:
                print(f'Loaded {bvals_masked.shape[0]} Masked B-Gradients on {unique_bvals_masked.shape[0]} shells: {unique_bvals_masked}')
        else:
            if self.verbose:
                print(f'No B-Mask provided')
            mask = np.ones(self.bvals.shape[0])
            bvals_masked = self.bvals[mask>0]
            bvecs_masked = self.bvecs[mask>0]
            unique_bvals_masked = np.unique(bvals_masked)
        return mask, bvals_masked, bvecs_masked, unique_bvals_masked

    def plot_bvecs(self, show_bvals=None):
        import plotly.graph_objects as go
        if show_bvals is None:
            show_bvals = self.unique_bvals
        for b in show_bvals:
            assert b in self.unique_bvals, f'B-Value {b} not found in B-Values {self.unique_bvals}'
            bvecs_b = self.bvecs[self.bvals==b]
            x = bvecs_b[:, 0]
            y = bvecs_b[:, 1]
            z = bvecs_b[:, 2]
            color = self.mask[self.bvals==b].astype(int)
            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=color, colorscale=[[0, 'rgb(128,177,211)'], [1, 'rgb(251,128,114)']]))])
            fig.update_layout(title=f'B-Gradients for B-Value {b}', scene_aspectmode='data')
            fig.show()

    def select_bvals(self, sel_bvals):
        if self.verbose:
            print(f'Selecting B-Values {sel_bvals} from {self.unique_bvals}')
        assert len(sel_bvals) > 0, f'Invalid selected B-Values {sel_bvals}'
        sel_bvals = np.unique(sel_bvals)
        assert sel_bvals[0] == 0, f'Invalid selected B-Values {sel_bvals} - First B-Value must be 0'
        mask = np.zeros(self.bvals.shape[0])
        for b in sel_bvals:
            if b in self.unique_bvals:
                mask[self.bvals==b] = 1
            else:
                if self.verbose:
                    print(f'B-Value {b} not found in B-Values {self.unique_bvals}')
        mask_masked_unique_bvals = np.zeros(self.unique_bvals_masked.shape[0], dtype=int)
        for i, b in enumerate(self.unique_bvals_masked):
            if b in sel_bvals:
                mask_masked_unique_bvals[i] = 1
        if self.verbose:
            print(f'Selected {mask_masked_unique_bvals} from masked B-Gradients')
        tmp_ = self.unique_bvals_masked[mask_masked_unique_bvals>0]
        index_bvals_sel_masked = np.zeros(tmp_.shape[0], dtype=int)
        for i, b in enumerate(tmp_):
            index_bvals_sel_masked[i] = sel_bvals.tolist().index(b)
        if self.verbose:
            print(f'Index {index_bvals_sel_masked} from masked B-Gradients')

        mask_unique_bvals = np.zeros(self.unique_bvals.shape[0], dtype=int)
        for i, b in enumerate(self.unique_bvals):
            if b in sel_bvals:
                mask_unique_bvals[i] = 1
        if self.verbose:
            print(f'Selected {mask_unique_bvals} from B-Gradients')
        tmp_ = self.unique_bvals[mask_unique_bvals>0]
        index_bvals_sel = np.zeros(tmp_.shape[0], dtype=int)
        for i, b in enumerate(tmp_):
            index_bvals_sel[i] = sel_bvals.tolist().index(b)
        if self.verbose:
            print(f'Index {index_bvals_sel} from masked B-Gradients')
        

        return mask, mask_masked_unique_bvals, index_bvals_sel_masked, mask_unique_bvals, index_bvals_sel
