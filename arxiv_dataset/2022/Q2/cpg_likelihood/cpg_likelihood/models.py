"""
Copyright (c) 2021 ghcollin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from . import priors
from . import coordinates
from . import llh

def default_unit_linear_prior(prior):
    if isinstance(prior, priors.Prior):
        return prior
    else:
        return priors.UnitLinearUniformPrior(*prior)


class Model(object):
    def __init__(self):
        pass

    def _get_params(self, name, out, thetas):
        raise NotImplementedError()

    def _add_model(self, model_list, poiss_map, thetas):
        raise NotImplementedError()


class NodeModel(Model):
    def __init__(self):
        super().__init__()

    def _add_model_F(self, model_list, poiss_map, F, thetas):
        raise NotImplementedError()


class ModelGroup(Model):
    def __init__(self, **models):
        super().__init__()
        self.models = models
        dtypes = [ (name, model.dtype) for name, model in models.items() ]
        self.dtype = np.dtype(dtypes)

    def _get_params(self, name_fn, val_fn, out, thetas):
        for name, model in self.models.items():
            model._get_params(name_fn, val_fn, out[name], thetas[name])

    def _add_model(self, model_list, poiss_map, thetas):
        for name, model in self.models.items():
            poiss_map = model._add_model(model_list, poiss_map, thetas[name])
        return poiss_map


class FluxRoot(Model):
    def __init__(self, F_prior, omega_node):
        super().__init__()
        self.F_prior = F_prior
        self.omega_node = omega_node
        dtypes = [ (self.F_prior.id(), float) ] + omega_node.dtypes
        self.dtype = np.dtype(dtypes)

    def _get_params(self, name_fn, val_fn, out, thetas):
        out[name_fn(self.F_prior)] = val_fn(self.F_prior, thetas[self.F_prior.id()])
        self.omega_node._get_params(name_fn, val_fn, out, thetas)

    def _add_model(self, model_list, poiss_map, thetas):
        Ftotal = self.F_prior.xform(thetas[self.F_prior.id()])
        return self.omega_node._add_model_F(model_list, poiss_map, Ftotal, thetas)


class OmegaNode(NodeModel):
    def __init__(self, omega_prior, model1_name, model1, model2_name, model2):
        super().__init__()
        self.omega_prior = default_unit_linear_prior(omega_prior)
        self.names = [model1_name, model2_name]
        self.models = [model1, model2]
        self.dtypes = [ (self.omega_prior.id(), float) ] \
             + [ (name, model.dtype) for name, model in zip(self.names, self.models) if model.dtype is not None ]
        self.dtype = np.dtype(self.dtypes)

    def _get_params(self, name_fn, val_fn, out, thetas):
        out[name_fn(self.omega_prior)] = val_fn(self.omega_prior, thetas[self.omega_prior.id()])
        _ = [ model._get_params(name_fn, val_fn, out[name], thetas[name]) for name, model in zip(self.names, self.models) if model.dtype is not None ]

    def _add_model_F(self, model_list, poiss_map, F, thetas):
        omega = self.omega_prior.xform(thetas[self.omega_prior.id()])
        return self.models[0]._add_model_F(
            model_list, 
            self.models[1]._add_model_F(
                model_list,
                poiss_map,
                (1-omega)*F,
                thetas[self.names[1]] if self.models[1].dtype is not None else None
            ),
            omega*F,
            thetas[self.names[0]] if self.models[0].dtype is not None else None
        )


class PoissLeaf(NodeModel):
    def __init__(self, eps_total):
        super().__init__()
        self.dtype = None
        self.eps_total = eps_total
        
    def _add_model_F(self, model_list, poiss_map, F, _):
        return poiss_map + F * self.eps_total


class NaturalPSLeaf(NodeModel):
    def __init__(self, eps, d_mu_eps, N_prior, beta_priors, n_priors):
        super().__init__()
        assert(len(n_priors) >= 2)
        assert(len(beta_priors) == len(n_priors)-2)

        beta_priors = list(map(default_unit_linear_prior, beta_priors))
        self.priors = [N_prior] + beta_priors + n_priors
        self.dtype = np.dtype([ (p.id(), float) for p in self.priors ])

        self.N_prior, self.n_priors, self.beta_priors = N_prior, n_priors, beta_priors
        self.eps, self.d_mu_eps = eps, d_mu_eps

    def _get_params(self, name_fn, val_fn, out, thetas):
        for p in self.priors:
            out[name_fn(p)] = val_fn(p, thetas[p.id()])

    def _add_model_F(self, model_list, poiss_map, F, thetas):
        N = self.N_prior.xform(thetas[self.N_prior.id()])
        betas = np.array([ p.xform(thetas[p.id()]) for p in self.beta_priors ])
        ns = np.array([ p.xform(thetas[p.id()]) for p in self.n_priors ])
        A, Fbs = coordinates.Ftot_betas_to_A_Fbs(N, F, betas, ns)
        
        model = llh.Model(A, Fbs, ns, self.eps, self.d_mu_eps)

        model_list.append(model)

        return poiss_map


class PoissModel(Model):
    def __init__(self, eps_total, F_prior):
        super().__init__()
        self.leaf = PoissLeaf(eps_total)
        self.F_prior = F_prior
        self.dtype = np.dtype([(F_prior.id(), float)])

    def _get_params(self, name_fn, val_fn, out, thetas):
        out[name_fn(self.F_prior)] = val_fn(self.F_prior, thetas[self.F_prior.id()])

    def _add_model(self, model_list, poiss_map, thetas):
        return self.leaf._add_model_F(model_list, poiss_map, self.F_prior.xform(thetas[self.F_prior.id()]), None)


class NaturalPSModel(Model):
    def __init__(self, eps, d_mu_eps, N_prior, F_total_prior, beta_priors, n_priors):
        super().__init__()
        self.leaf = NaturalPSLeaf(eps, d_mu_eps, N_prior, beta_priors, n_priors)
        self.F_prior = F_total_prior
        self.dtype = np.dtype([(F_total_prior.id(), float)] + self.leaf.dtype.descr)

    def _get_params(self, name_fn, val_fn, out, thetas):
        out[name_fn(self.F_prior)] = val_fn(self.F_prior, thetas[self.F_prior.id()])
        self.leaf._get_params(name_fn, val_fn, out, thetas)

    def _add_model(self, model_list, poiss_map, thetas):
        return self.leaf._add_model_F(model_list, poiss_map, self.F_prior.xform(thetas[self.F_prior.id()]), thetas)


class StandardPSModel(Model):
    def __init__(self, eps, d_mu_eps, A_prior, Fb_priors, n_priors):
        assert(len(n_priors) >= 2)
        assert(len(Fb_priors) == len(n_priors)-1)

        self.priors = [A_prior] + Fb_priors + n_priors
        self.dtype = np.dtype([ (p.id(), float) for p in self.priors ])

        self.A_prior, self.n_priors, self.Fb_priors = A_prior, n_priors, Fb_priors
        self.eps, self.d_mu_eps = eps, d_mu_eps

    def _get_params(self, name_fn, val_fn, out, thetas):
        for p in self.priors:
            out[name_fn(p)] = val_fn(p, thetas[p.id()])

    def _add_model(self, model_list, poiss_map, thetas):
        A = self.A_prior.xform(thetas[self.A_prior.id()])
        Fbs = np.array([ p.xform(thetas[p.id()]) for p in self.Fb_priors ])
        ns = np.array([ p.xform(thetas[p.id()]) for p in self.n_priors ])

        model = llh.Model(A, Fbs, ns, self.eps, self.d_mu_eps)

        model_list.append(model)

        return poiss_map

def flatten_dtype(dtype):
    def flatten_desc(root, desc):
        return [ (namei, ti) for nameo, to in desc for namei, ti in (flatten_desc(root + nameo + '/', to) if isinstance(to, list) else [(root + nameo, to)]) ]

    return np.dtype(flatten_desc('', dtype.descr))


import collections
recursive_dict = lambda: collections.defaultdict(recursive_dict)


def flatten_dict_to_pairs(d):
    return [ (ki, vi) for ko, vo in d.items() for ki, vi in (flatten_dict_to_pairs(vo) if isinstance(vo, dict) else [(ko, vo)]) ]


def unique_list(l):
    unique = set()
    for e in l:
        if e in unique:
            return False
        else:
            unique.add(e)
    return True

import threading

class JointDistribution(object):
    def __init__(self, data, epsilons, model, threads = 1):
        self.model = model
        self.flat_dtype = flatten_dtype(self.model.dtype)
        self.total_params = len(self.flat_dtype.descr)
        self.empty_map = np.zeros_like(data).astype(float)

        self.workspace = (data, len(epsilons), threads)
        self.tl = threading.local()

    def __getstate__(self):
        self.tl = None
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tl = threading.local()

    def get_params(self, thetas):
        result = np.squeeze(thetas.copy().view(self.model.dtype))
        self.model._get_params(lambda p: p.id(), lambda p, t: p.xform(t), result, result)
        return result

    def get_params_flat(self, thetas, display_value=False, display_name=False):
        result = recursive_dict()
        name_fn = (lambda p: p.display()) if display_name else (lambda p: p.id())
        val_fn = (lambda p, t: p.display_value(t)) if display_value else (lambda p, t: p.xform(t))
        self.model._get_params(name_fn, val_fn, result, np.squeeze(thetas.view(self.model.dtype)))
        flat = flatten_dict_to_pairs(result)
        keys, vals = zip(*flat)
        if not unique_list(keys):
            raise Exception("Collision in flat parameter names, make sure all parameter names are unique.")
        return dict(zip(keys, vals))

    def get_id_display_map(self):
        result = recursive_dict()
        self.model._get_params(lambda p: p.id(), lambda p, _: p.display(), result, result)
        flat = flatten_dict_to_pairs(result)
        keys, vals = zip(*flat)
        if not unique_list(keys):
            raise Exception("Collision in flat parameter names, make sure all parameter names are unique.")
        return dict(zip(keys, vals))


    def ln_p(self, thetas, return_bad=False):
        thetas_ary = thetas.view(self.model.dtype)

        workspace = self.tl.__dict__.setdefault('workspace', llh.Workspace(*self.workspace))

        #print((self.flat_dtype.itemsize, self.model.dtype.itemsize, self.total_params, self.flat_dtype.descr))
        vals = []
        bad = False
        for thetas in thetas_ary:
            ps_models = []
            poiss_model = self.empty_map
            poiss_model = self.model._add_model(ps_models, poiss_model, thetas)

            assert(len(ps_models) > 0)
            assert(poiss_model is not None)
            val, bad_ = workspace.eval(ps_models, poiss_model)
            vals.append(val)
            bad = bad or bad_

        return (np.squeeze(vals), bad) if return_bad else val
