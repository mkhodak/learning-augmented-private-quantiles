import pdb
from copy import deepcopy
from functools import lru_cache
from operator import itemgetter
import numpy as np
from scipy import special
from prior import GLOBAL_MAX, GLOBAL_MIN


def exact_quantiles(x, quantiles, interval=1.):
    n = len(x)
    bounds, idx = np.unique(x, return_index=True)
    out = []
    i = 1
    for q in quantiles:
        k = int(q*n)
        while True:
            try:
                if k < idx[i]:
                    out.append(bounds[i-1] if k-idx[i-1] < idx[i]-k else bounds[i])
                    break
            except IndexError:
                out.append(bounds[-1] + interval)
                break
            i += 1
    return out


def exact_intervals(x, quantiles, interval=1.):
    n = len(x)
    bounds, idx = np.unique(x, return_index=True)
    a, b = np.empty(len(quantiles)), np.empty(len(quantiles))
    i = 1
    for j, q in enumerate(quantiles):
        k = int(q*n)
        while True:
            try:
                if k < idx[i]:
                    if k-idx[i-1] < idx[i]-k:
                        a[j] = bounds[i-2] if i > 1 else bounds[0]-interval
                        b[j] = bounds[i-1]
                    else:
                        a[j] = bounds[i-1]
                        b[j] = bounds[i]
                    break
            except IndexError:
                a[j] = bounds[i-1]
                b[j] = bounds[i-1] + interval
                break
            i += 1
    return a, b


def gaps(x, quantiles, estimates):
    n = len(x)
    bounds, idx = np.unique(x, return_index=True)
    bounds = np.append(np.append(-float('inf'), bounds), float('inf'))
    out = []
    i = 1
    for q, estimate in zip(quantiles, estimates):
        while True:
            if estimate <= bounds[i]:
                k = int(q*n)
                g = np.append(np.absolute(k - idx), n-k)
                g -= g.min()
                out.append(g[i-1])
                break
            i += 1
    return out


@lru_cache(maxsize=None)
def quantile_tree(qs, epsilon, K=2, p=0.):
    depths = int(np.ceil(np.log(len(qs)+1) / np.log(K)))
    epsbar = epsilon / sum(1. / (k+1)**p for k in range(depths))
    L = [{} for _ in range(depths)]
    def recurse(quantiles, i, qa, qb, budget):
        if len(quantiles) < K:
            for q in quantiles:
                L[i][q] = {'qa': qa, 'qb': qb, 'epsilon': budget / len(quantiles)}
        else:
            idx = [int(np.ceil(j*len(quantiles) / K))-1 for j in range(1, K)]
            qaj = qa
            ij = 0
            for j in idx:
                children = quantiles[ij:j]
                L[i][quantiles[j]] = {'qa': qa, 'qb': qb, 'epsilon': epsbar / (i+1)**p / len(idx)}
                if children:
                    recurse(children, i+1, qaj, (i, quantiles[j]), budget - epsbar / (i+1)**p)
                qaj = (i, quantiles[j])
                ij = j+1
            children = quantiles[ij:]
            if children:
                recurse(children, i+1, qaj, qb, budget - epsbar / (i+1)**p)
    recurse(qs, 0, (-1, 0.), (-1, 1.), epsilon)
    return L


def multiple_quantiles(x, qs, epsilon, priors, edge=False, K=2, p=0.):

    n = len(x)
    bounds, idx = np.unique(x, return_index=True)
    bounds = np.append(np.append(GLOBAL_MIN, bounds), GLOBAL_MAX)
    L = deepcopy(quantile_tree(qs, epsilon, K=K, p=p))
    L.append({0.: {'out': GLOBAL_MIN, 'idx': 0}, 1.: {'out': GLOBAL_MAX, 'idx': len(bounds)-1}})

    for depth in range(len(L)-1):

        if K > 2:
            initial_releases = []
            quantile_utils = []

        for q, node in sorted(L[depth].items(), key=itemgetter(0)):

            prior = priors[qs.index(q)]
            k = int(q*n)
            utils = -np.append(np.absolute(k - idx), n-k)
            utils -= utils.max()
            with np.errstate(under='ignore'):
                probs = np.diff(prior.cdf(bounds))
                scores = np.exp(.5 * node['epsilon'] * utils)
                weights = scores * probs
            node['Psi'] = weights.sum()
            node['Psi*'] = weights[utils == 0.].sum()

            qa, qb = node['qa'], node['qb']
            a, b = L[qa[0]][qa[1]], L[qb[0]][qb[1]]
            aidx, bidx = a['idx'], b['idx']
            aout, bout = a['out'], b['out']
            if aout > GLOBAL_MIN or bout < GLOBAL_MAX:
                if edge:
                    if aidx:
                        edge_a = probs[:aidx].sum()
                        probs[aidx] += edge_a
                        weights[:aidx] = 0.
                        with np.errstate(under='ignore'):
                            weights[aidx] = scores[aidx] * probs[aidx]
                    if bidx < len(bounds)-1:
                        edge_b = probs[bidx+1:].sum()
                        probs[bidx] += edge_b
                        weights[bidx+1:] = 0.
                        with np.errstate(under='ignore'):
                            weights[bidx] = scores[bidx] * probs[bidx]
                else:
                    with np.errstate(under='ignore'):
                        probs = np.diff(prior.cdf(bounds, a=aout, b=bout))
                        weights = scores * probs
                node['PsiHat'] = weights.sum()
            else:
                node['PsiHat']= node['Psi']
            with np.errstate(under='ignore'):
                weights /= node['PsiHat']
                node['E[Gap]'] = -sum(utils * weights)
                k = np.random.choice(len(weights), p=weights)

            if edge and k == aidx and aout > GLOBAL_MIN:
                edge_a += prior.cdf(bounds[k+1]) - prior.cdf(aout)
            else:
                edge_a = 0.
            if edge and k == bidx and bout < GLOBAL_MAX:
                edge_b += prior.cdf(bout) - prior.cdf(bounds[k])
            else:
                edge_b = 0.
            r = np.random.rand()
            if r < edge_a:
                sample = aout
            elif r > 1.-edge_b:
                sample = bout
            else:
                sample = prior.its(max(bounds[k], aout), min(bounds[k+1], bout))

            if K > 2:
                initial_releases.append((sample, k))
                quantile_utils.append((q, utils))
            else:
                node['idx'] = k
                node['Gap'] = -utils[k]
                node['out'] = sample
                node['dist'] = abs(sample - np.quantile(x, q))

        if K > 2:
            for (sample, k), (q, utils) in zip(sorted(initial_releases, key=itemgetter(0)), quantile_utils):
                node = L[depth][q]
                node['idx'] = k
                node['Gap'] = -utils[k]
                node['out'] = sample
                node['dist'] = abs(sample - np.quantile(x, q))

    return [results for (_, results) in sorted((q for depth in L[:-1] for q in depth.items()), key=itemgetter(0))]
