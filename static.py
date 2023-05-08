import os
import pdb
import random
import sys
import time
import traceback
from collections import namedtuple, OrderedDict
import h5py
import numpy as np; np.seterr(all='raise')
from mpi4py import MPI
from scipy import stats
from data import Adult, Goodreads, Gaussian
from parallel import dump, write
from prior import Cauchy, HalfCauchy, Uniform
from quantiles import multiple_quantiles


Metrics = namedtuple('Metrics', ['avgGap', 'maxGap', 'negLogPsi', 'negLogPsiStar', 'negLogPsiHat', 'dist', 'expGap'])


def main():
    
    comm = MPI.COMM_WORLD
    runs = OrderedDict()
    for n in [100, 1000, 10000, 100000]:
        for data, subset in [
                             (Adult, 'train'),
                             (Goodreads, 'poetry'),
                             (Gaussian, '0.0'),
                             (Gaussian, '4.0'),
                             ]:
            kwargs = {} if data.name == 'Gaussian' else {'verbose': not comm.rank}
            for name, x in data.data(**kwargs)[subset].items():
                np.random.seed(comm.rank)
                x = np.sort(np.random.choice(x, n, replace=False))
                fname = os.path.join('dump', f'{data.name}_{subset if name is None else name}.h5')
                for logeps in np.linspace(-3., 1., 9):
                    for dist, prior in [
                                        ('Uniform', lambda q, guesses: Uniform(*guesses.qbound(q))),
                                        ('Cauchy', lambda q, guesses: Cauchy(guesses.loc, guesses.scale)),
                                        ]:
                        for edge in [False, True]:
                            if n == 100000 and not edge:
                                continue
                            for K in range(2, 9):
                                for p in [0., 1.5, 2.]:
                                    if n == 100000 and p != 2.:
                                        continue
                                    path = os.path.join(f'n={n}', f'logeps={logeps}', dist, 'edge' if edge else 'cond', f'K={K}', f'p={p}')
                                    if os.path.isfile(fname):
                                        with h5py.File(fname, 'r') as f:
                                            if path in f and len(f[path]):
                                                if not comm.rank:
                                                    write('skipping', fname, path)
                                                continue
                                    runs[(fname, path)] = (data, name, x, 10.**logeps, prior, edge, K, p)

    for (fname, path), (data, name, x, epsilon, prior, edge, K, p) in runs.items():
        metrics = Metrics._make(np.append(np.nan*np.ones(K-2), np.empty(101-K)) for _ in range(7))
        for m in range(K-1, 100):
            np.random.seed(comm.rank)
            random.seed(comm.rank)
            qs = tuple((i+1)/(m+1) for i in range(m))
            try:
                results = multiple_quantiles(x, qs, epsilon, [prior(q, data.guesses[name]) for q in qs], edge=edge, K=K, p=p)
                metrics.avgGap[m-1] = sum(data['Gap'] for data in results) / len(qs)
                metrics.maxGap[m-1] = max(data['Gap'] for data in results)
                metrics.negLogPsi[m-1] = -np.log(stats.hmean([data['Psi'] for data in results]))
                with np.errstate(divide='ignore'):
                    metrics.negLogPsiStar[m-1] = -np.log(stats.hmean([data['Psi*'] for data in results]))
                metrics.negLogPsiHat[m-1] = -np.log(stats.hmean([data['PsiHat'] for data in results]))
                metrics.dist[m-1] = np.median([data['dist'] for data in results])
                metrics.expGap[m-1] = sum(data['E[Gap]'] for data in results) / len(qs)
            except Exception as e:
                print('process', comm.rank, 'failed:', e)
                traceback.print_tb(e.__traceback__, file=sys.stdout)
                sys.stdout.flush()
                sys.exit()
        dump(metrics, fname, path)


if __name__ == '__main__':

    main()
