import os
import pdb
import pickle
import random
import sys
import time
import traceback
import warnings
from collections import namedtuple
from itertools import product
from operator import itemgetter
import numpy as np; np.seterr(all='raise')
import torch; torch.set_default_tensor_type(torch.DoubleTensor); torch.use_deterministic_algorithms(True)
from mpi4py import MPI
from scipy import stats
from torch import nn, optim
from cocob import COCOBBackprop
from data import Adult, Goodreads
from learn import censored_l3, ReparameterizedLinear
from parallel import dump, write
from prior import Cauchy, HalfCauchy, Laplace, Mixture, Uniform, ROBUST_PRIOR, ROBUST_COEF
from quantiles import exact_intervals, exact_quantiles, gaps, multiple_quantiles, quantile_tree


def fit_laplace(n, m, de, theta=0., phi=1., theta_lr=1., phi_lr=.1, interval=0., max_iter=1000, batchsize=1000, tolerance=.001, phi_lower=.01, opt=optim.SGD, verbose=False, bootstrap=False):

    qs = tuple((i+1)/(m+1) for i in range(m))
    with warnings.catch_warnings(record=True):
        model = ReparameterizedLinear(0, m, theta=theta, phi=phi)
    optimizer = opt(model.parameters())
    losses = []
    prev = float('inf')
    f = torch.empty((1, 0))
    for i in range(max_iter):
        if bootstrap:
            x = np.sort(np.random.choice(de, (batchsize, n)))
        else:
            x = np.sort(np.random.choice(de, (len(de)//n, n), replace=False))
        for j in range(len(x)):
            optimizer.zero_grad()
            a, b = exact_intervals(x[j], qs, interval=interval)
            loss = torch.logsumexp(censored_l3(torch.Tensor(a), torch.Tensor(b), *model(f))[0], 0)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            model.project(phi_lower)
        avg_loss = np.mean(losses)
        if verbose:
            write('iter', i, 'loss:', round(avg_loss, 4))
        if abs(prev - avg_loss) < tolerance:
            break
        prev = avg_loss
        if not bootstrap:
            break

    locs, scales = model.predict(f)
    return locs.detach().numpy(), scales.detach().numpy(), np.mean(losses[-batchsize:])


Metrics = namedtuple('Metrics', ['avgGap', 'maxGap'])


def main():

    comm = MPI.COMM_WORLD
    i = 0
    for n in [10 ** j for j in range(2, 4)]:
        for m in [1, 3, 4, 9, 19, 29]:
            for dataset, sources in [
                                     (Adult, ['train']),
                                     (Goodreads, ['history', 'mystery']),
                                     ]:
                data = dataset.data()
                for source in sources:
                    for key, x in data[source].items():
                        g = dataset.guesses[key]
                        mask = np.ones(len(x), dtype=np.bool)
                        if source != 'train':
                            mask[np.random.choice(len(x), 10000, replace=False)] = False
                        path = os.path.join('dump', f'{dataset.name}-{source}-{key}-n{n}-m{m}_Non.pkl')
                        if not os.path.isfile(path):
                            if comm.rank == i % comm.size:
                                np.random.seed(comm.rank)
                                random.seed(comm.rank)
                                torch.manual_seed(comm.rank)
                                locs, scales, loss = fit_laplace(n, m, x[mask], theta=g.loc/g.scale, phi=1./g.scale, interval=g.interval, opt=COCOBBackprop, bootstrap=False)
                                with open(path, 'wb') as f:
                                    pickle.dump({'locs': locs, 'scales': scales, 'mask': mask, 'loss': loss}, f)
                                write('process', comm.rank, 'completed', path, 'with loss', round(loss, 2))
                            i += 1


    for dataset, source, target in [
                                    (Adult, 'train', 'test'),
                                    (Goodreads, 'history', 'poetry'),
                                    ]:
        data = dataset.data()
        for key, x_tgt in data[target].items():
            g = dataset.guesses[key]
            for n in [100, 1000]:
                for m in [9]:
                    np.random.seed(comm.rank)
                    random.seed(comm.rank)
                    qs = tuple((i+1)/(m+1) for i in range(m))
                    uniform = [Uniform(*g.qbound(q)) for q in qs]
                    halfcauchy = [HalfCauchy(0., g.loc) for q in qs]
                    path = os.path.join('dump', f'{dataset.name}-{source}-{key}-n{n}-m{m}_Non.pkl')
                    with open(path, 'rb') as f:
                        def_model = pickle.load(f)
                    defit = [Laplace(loc, scale) for loc, scale in zip(def_model['locs'], def_model['scales'])]
                    x_src = data[source][key]
                    mask = def_model['mask']
                    x_src, x_mix = x_src[mask], x_src[~mask]
                    if source == 'train':
                        x_mix = x_tgt
                    q_src = exact_quantiles(x_src, qs, interval=g.interval)
                    cauchy = [Cauchy(q, g.scale) for q in q_src]
                    for name, priors in [
                                         ('Uniform', uniform),
                                         ('HalfCauchy', halfcauchy),
                                         ('DEFit', [Mixture(prior, ROBUST_PRIOR, ROBUST_COEF) for prior in defit]),
                                         ('DEFit-robust', [Mixture(prior1, prior2, .1) for prior1, prior2 in zip(defit, halfcauchy)]),
                                         ('public-quantiles', None),
                                         ('public-Cauchy', cauchy),
                                         ]:
                        for logeps in np.linspace(-3., 1., 9):
                            k = 1 if source == 'train' else 11
                            metrics = Metrics(np.empty(k), np.empty(k))
                            for i, mix in enumerate(np.linspace(0., 1., k)):
                                x = np.sort(np.append(np.random.choice(x_tgt, int(mix*n), replace=False),
                                                      np.random.choice(x_mix, n-int(mix*n), replace=False)))
                                if priors is None:
                                    results = gaps(x, qs, q_src)
                                elif type(priors) == dict:
                                    results = [data['Gap'] for data in multiple_quantiles(x, qs, 10.**logeps, priors[logeps])]
                                else:
                                    results = [data['Gap'] for data in multiple_quantiles(x, qs, 10.**logeps, priors)]
                                metrics.avgGap[i] = sum(results) / m
                                metrics.maxGap[i] = max(results)
                            dump(metrics, f'dump/{target}-pubpri_Non.h5', os.path.join(path[path.index('-')+1:path.index('_Non')], f'logeps={logeps}', name))


if __name__ == '__main__':

    main()
