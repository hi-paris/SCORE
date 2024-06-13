import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern, RBF
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

rand_seed = 0
np.random.seed(rand_seed)
random.seed(rand_seed)

def latin_hypercube(n_pts:int, dim:int):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]
    # Add some perturbations within each box
    return X + np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)

def from_unit_cube(x:float, lb:float, ub:float):
    return x * (ub - lb) + lb

def acq_fun(af: str, mean: float, std: float, fX_best: float, beta: float=0, xi: float=0):
    if af == 'LCB':
        return mean - beta * std
    elif af == 'PI':
        z = (fX_best - mean - xi) / (std + 1e-16)  # for minimization
        return -norm.cdf(z)  # because we sort by ascending AF values
    elif af == 'EI':
        z = (fX_best - mean - xi) / (std + 1e-16)
        return - ((fX_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z))
    else:
        return mean - beta * std

class SCORE:
    def __init__(self, parameters, f, n_init:float=0, init_combs=None, af='EI', beta=0, xi=0, gp=None):
        Add_combs=False
        if init_combs is not None:
            init_combs = list(init_combs)
            if n_init > 0:
                Add_combs=True
        else:
            Add_combs=True
            if n_init == 0:
                n_init = len(parameters)*2
            init_combs=[]
        if Add_combs:
            init_combs_tmp=[]
            X_init = np.transpose(latin_hypercube(n_init, len(parameters)))
            for p, param in enumerate(parameters):
                param_range=param['domain']
                if param['scale'] == 'log':
                    X_init_param = from_unit_cube(X_init[p], np.min(np.log10(param_range)), np.max(np.log10(param_range)))
                    X_init_param = 10 ** X_init_param
                else:
                    X_init_param = from_unit_cube(X_init[p], np.min(param_range), np.max(param_range))
                X_init_param = [min(param_range, key=lambda x: abs(x - item)) for item in X_init_param]  # take closest value from param_range
                init_combs_tmp.append(X_init_param)
            init_combs.extend(list(zip(*init_combs_tmp)))
            
        if gp==None:
            gp = GaussianProcessRegressor(ConstantKernel(1.0, constant_value_bounds='fixed') * Matern(len(parameters), length_scale_bounds='fixed'))
            
        self.parameters = parameters
        self.f = f
        self.gp = gp
        self.init_combs = init_combs
        self.af = af
        self.beta = beta
        self.xi = xi

    def fit(self, nb_it, n_cbs=1, verbose=True):
        param_names=[param['name'] for param in self.parameters]
        bo = pd.DataFrame(self.init_combs, columns=param_names)
        bo['obj_func'] = bo[param_names].apply(self.f, axis=1)
        min_target = [bo['obj_func'].min()]

        if verbose:
            print(0, len(bo))
            print(bo.sort_values('obj_func')[['obj_func']].head())

        score_cols = [param['name'] + '_score' for param in self.parameters]
        for bo_it in range(1, nb_it + 1):
            top_param_vals = pd.DataFrame()
            for p, param in enumerate(self.parameters):
                param_range = param['domain']
                bo_param = bo.groupby(param['name']).min('obj_func')
                ind = list(bo_param.index)
                ind.extend(param_range)
                bo_param = bo_param.reindex(sorted(set(ind)))

                if param['scale'] == 'log':
                    self.gp.fit(np.array(np.log10(bo_param['obj_func'].dropna().index)).reshape(-1, 1), (bo_param['obj_func'].dropna()).values)
                    bo_param[['Mean', 'STD']] = pd.DataFrame(self.gp.predict(np.array(np.log10(bo_param.index)).reshape(-1, 1), return_std=True)).T.values
                else:
                    self.gp.fit(np.array(bo_param['obj_func'].dropna().index).reshape(-1, 1), (bo_param['obj_func'].dropna()).values)
                    bo_param[['Mean', 'STD']] = pd.DataFrame(self.gp.predict(np.array(bo_param.index).reshape(-1, 1), return_std=True)).T.values

                fX_best = bo_param['obj_func'].min()
                mean, std = bo_param['Mean'], bo_param['STD']

                bo_param['AF'] = acq_fun(self.af, mean, std, fX_best, self.beta, self.xi)

                bo_param = bo_param.sort_values('AF', ascending=True)

                top_param_vals[param['name']] = bo_param.index
                top_param_vals[param['name']+'_score'] = bo_param['AF'].values

            # Get most likely combinations using most likely values for each param
            top_param_vals['Score'] = top_param_vals[score_cols].apply(np.sum, axis=1)

            # Remove duplicated predictions
            combs_extra = pd.concat([top_param_vals[param_names+['Score']], bo[param_names]]).reset_index(drop=True)
            combs_extra = combs_extra.drop_duplicates(param_names, keep=False)
            combs_extra = combs_extra.sort_values('Score', ascending=True).reset_index(drop=True)

            # Compute f(x) of n_cbs new combinations
            bo = pd.concat([bo, combs_extra.loc[:n_cbs - 1]]).reset_index(drop=True)
            n_new = bo['obj_func'].dropna().index[-1] + 1
            bo.loc[n_new:, 'obj_func'] = bo.loc[n_new:, param_names].apply(self.f, axis=1)

            min_target.append(bo['obj_func'].min())
            if verbose: # Print top 5 parameter combinations after each iteration
                print(bo_it, len(bo))
                print(bo.sort_values('obj_func', ascending=True)[['obj_func']].head())

        self.nb_it = nb_it
        self.n_cbs = n_cbs
        self.bo = bo
        self.min_target = min_target

        return min(self.min_target), self.bo

    def plot(self):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(np.multiply(range(self.nb_it+1), self.n_cbs), self.min_target, '-o')
        ax.set_ylabel('Minimum Value Found')
        ax.set_xlabel('Number of Function Evaluations')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    def return_min(self)->float:
        return self.bo[self.bo['obj_func']==min(self.min_target)].iloc[0]










