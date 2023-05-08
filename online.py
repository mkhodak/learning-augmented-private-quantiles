import os
import pdb
import random
import sys
import time
import traceback
from collections import namedtuple, OrderedDict
from copy import deepcopy
from functools import lru_cache
import h5py
import numpy as np; np.seterr(all='raise')
import torch; torch.set_default_tensor_type(torch.DoubleTensor); torch.use_deterministic_algorithms(True)
from mpi4py import MPI
from scipy import stats
from torch import optim
from cocob.cocob import COCOBBackprop
from data import CitiBike, Synthetic, Flesch
from learn import log_laplace_integral, ReparameterizedLinear, censored_l3, DPFTRL
from parallel import dump, write
from prior import Cauchy, HalfCauchy, Laplace, Mixture, Uniform, GLOBAL_MIN, GLOBAL_MAX, ROBUST_PRIOR, ROBUST_COEF
from quantiles import exact_intervals, multiple_quantiles


Metrics = namedtuple('Metrics', ['avgGap', 'maxGap', 'negLogPsi', 'negLogPsiStar', 'negLogPsiHat', 'loss', 'dist', 'expGap'])


def evaluate(method, X, F, qs, epsilon, verbose=True, **kwargs):
    metrics = Metrics._make(np.empty(len(X)) for _ in range(8))
    for i, (x, f) in enumerate(zip(X, F)):
        if verbose and not (i+1) % 100:
            write('day', i+1, end='\r')
        priors = method.predict(f)
        results = multiple_quantiles(x, qs, epsilon, priors, **kwargs)
        loss = method.update(x, f, [data['out'] for data in results])
        metrics.avgGap[i] = sum(data['Gap'] for data in results) / len(qs)
        metrics.maxGap[i] = max(data['Gap'] for data in results)
        metrics.negLogPsi[i] = -np.log(stats.hmean([data['Psi'] for data in results]))
        with np.errstate(divide='ignore'):
            metrics.negLogPsiStar[i] = -np.log(stats.hmean([data['Psi*'] for data in results]))
        metrics.negLogPsiHat[i] = -np.log(stats.hmean([data['PsiHat'] for data in results]))
        metrics.loss[i] = loss
        metrics.dist[i] = np.median([data['dist'] for data in results])
        metrics.expGap[i] = sum(data['E[Gap]'] for data in results) / len(qs)
    return metrics


class Fixed:
    def __init__(self, priors):
        self.priors = priors
    def predict(self, f):
        return self.priors
    def update(self, x, f, r):
        return float('nan')


class PrevRelease:
    def __init__(self, locs, scales, robust_prior=ROBUST_PRIOR, robust_coef=ROBUST_COEF):
        self.priors = [Mixture(Laplace(loc, scale), robust_prior, robust_coef)
                       for loc, scale in zip(locs, scales)]
    def predict(self, f):
        return self.priors
    def update(self, x, f, r):
        for prior, loc in zip(self.priors, r):
            prior.prior1.loc = loc
        return float('nan')


class LearnedPriors:

    def __init__(self, n_features, quantiles, lr_theta=.1, lr_phi=.001, phi_lower=.01, robust_prior=ROBUST_PRIOR, robust_coef=ROBUST_COEF, opt=optim.SGD, **kwargs):
        self.model = ReparameterizedLinear(n_features, len(quantiles), **kwargs)
        try:
            self.optimizer = opt([{'params': self.model.linear.parameters()},
                                  {'params': self.model.phi.parameters(), 'lr': lr_phi}],
                                  lr=lr_theta)
        except TypeError:
            self.optimizer = opt(self.model.parameters())
        self.phi_lower = phi_lower
        self.priors = [Mixture(Laplace(0., 1.), robust_prior, robust_coef)
                       for _ in quantiles]
        self.quantiles = quantiles

    def predict(self, f):
        for prior, loc, scale in zip(self.priors, *self.model.predict(f)):
            prior.prior1.loc = loc.item()
            prior.prior1.scale = scale.item()
        return self.priors

    def update(self, x, f, r):
        self.optimizer.zero_grad()
        a, b = exact_intervals(x, self.quantiles)
        loss = torch.logsumexp(censored_l3(torch.Tensor(a), torch.Tensor(b), *self.model(f)), 0)
        loss.backward()
        self.optimizer.step()
        self.model.project(self.phi_lower)
        return loss.item()


def proxy_l3(proxy_location, interval_radius, theta, phi):
    offset = torch.absolute(theta - phi * proxy_location)
    radius = phi * interval_radius
    return log_laplace_integral(torch.absolute(theta - phi * proxy_location),
                                phi * interval_radius)


class ProxyLearn(LearnedPriors):

    def __init__(self, n_features, quantiles, radius=1E-8, nucnorm=0., **kwargs):
        super().__init__(n_features, quantiles, **kwargs)
        self.radius = radius
        self.nucnorm = nucnorm

    def update(self, x, f, r):
        r = torch.Tensor(r)
        self.optimizer.zero_grad()
        theta, phi = self.model(f)
        if self.nucnorm:
            loss = torch.logsumexp(proxy_l3(r, self.radius, *self.model(f)), 0) + self.nucnorm * torch.norm(self.model.linear.weight, 'nuc')
        else:
            loss = torch.logsumexp(proxy_l3(r, self.radius, *self.model(f)), 0)
        loss.backward()
        self.optimizer.step()
        self.model.project(self.phi_lower)
        return loss.item()


class DPLearn(LearnedPriors):

    def __init__(self, n_features, quantiles, T, epsilon, lr_theta=.1, lr_phi=.001, clip=1.,  **kwargs):
        super().__init__(n_features, quantiles, **kwargs)
        self.opt_theta = DPFTRL(self.model.linear, lr_theta, .5*epsilon, .1/T, T, 1, clip)
        self.opt_phi = DPFTRL(self.model.phi, lr_phi, .5*epsilon, .1/T, T, 1, clip)
        self.clip = clip

    def update(self, x, f, r):
        self.opt_theta.zero_grad()
        self.opt_phi.zero_grad()
        theta, phi = self.model(f)
        a, b = exact_intervals(x, self.quantiles)
        loss = torch.logsumexp(censored_l3(torch.Tensor(a), torch.Tensor(b), theta, phi), 0)
        loss.backward()
        with torch.no_grad():
            for model in [self.model.linear, self.model.phi]:
                norm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters()))
                if norm > self.clip:
                    for p in model.parameters():
                        p.grad *= self.clip / norm
        self.opt_theta.step()
        self.opt_phi.step()
        self.model.project(self.phi_lower)
        return loss.item()


def main():

    comm = MPI.COMM_WORLD
    runs = OrderedDict()
    for dataset, setting, kwargs in [
                                     (CitiBike, '', {}),
                                     (Synthetic, '_0.0_T1E5', {'T': 100000, 'rate': 0.0})
                                     (Synthetic, '_0.0', {'rate': 0.0}),
                                     (Synthetic, '_1.0', {'rate': 1.0}),
                                     (Synthetic, '_5.0', {'rate': 5.0}),
                                     (Flesch, '', {}),
                                     ]:
        fname = os.path.join('dump', f'{dataset.name}{setting}.h5')
        g = dataset.guesses
        for m in [1, 9]:
            X, F = dataset.features(n_quantiles=m, **kwargs)
            qs = tuple((i+1)/(m+1) for i in range(m))
            for logeps in np.append(np.linspace(-4., -1., 7), np.linspace(-.75, 1., 8)):
            for logeps in [-1.]:
                robust = {'robust_prior': HalfCauchy(0., g.loc) if g.nonnegative else Cauchy(g.loc, g.scale), 
                          'robust_coef': .1}
                init = {
                        'theta': g.loc/g.scale,
                        'phi': 1./g.scale,
                        }
                for name, method in [
                                     ('Uniform', Fixed([Uniform(*g.qbound(q)) for q in qs])),
                                     ('Cauchy', Fixed([Cauchy(g.loc, g.scale)]*m)),
                                     ('HalfCauchy', Fixed([HalfCauchy(0., g.loc)]*m)),
                                     ('PrevRelease', PrevRelease([g.loc]*m, [g.scale]*m)),
                                     ('PrevRelease-robust', PrevRelease([g.loc]*m, [g.scale]*m, **robust)),
                                     ('COCOB-robust', LearnedPriors(len(F[0]), qs, **init, opt=COCOBBackprop, **robust)),
                                     ('ProxyCOCOB', ProxyLearn(len(F[0]), qs, **init, radius=.5*g.interval, opt=COCOBBackprop)),
                                     ('ProxyCOCOB-robust', ProxyLearn(len(F[0]), qs, **init, radius=.5*g.interval, opt=COCOBBackprop, **robust)),
                                     ('DPFTRL', DPLearn(len(F[0]), qs, len(X), .1*10.**logeps, **init, lr_theta=m*.0001, lr_phi=m*.00001)),
                                     ('DPFTRL-robust', DPLearn(len(F[0]), qs, len(X), .1*10.**logeps, **init, lr_theta=m*.0001, lr_phi=m*.00001, **robust)),
                                     ]:
                    if name == 'HalfCauchy' and not g.nonnegative:
                        continue
                    path = os.path.join(name, f'logeps={logeps}', f'm={m}')
                    if os.path.isfile(fname):
                        with h5py.File(fname, 'r') as f:
                            if path in f and len(f[path]):
                                if not comm.rank:
                                    write('skipping', path)
                                continue
                    runs[(fname, path)] = (X, F, (.9 if 'DPFTRL' in path else 1.)*10. ** logeps, qs, method)

    for (fname, path), (X, F, epsilon, qs, method) in runs.items():
        np.random.seed(comm.rank)
        random.seed(comm.rank)
        torch.manual_seed(comm.rank)
        try:
            metrics = evaluate(method, X, F, qs, epsilon, verbose=not comm.rank, edge='edge' in path)
        except Exception as e:
            print('process', comm.rank, 'failed:', e)
            traceback.print_tb(e.__traceback__, file=sys.stdout)
            sys.stdout.flush()
            sys.exit()
        dump(metrics, fname, path)
        

if __name__ == '__main__':

    main()
