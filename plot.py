import os
import pdb
import h5py
import numpy as np
from matplotlib import pyplot as plt
from data import CitiBike, Synthetic


LEGEND = {'fontsize': 16}
LABELS = {'fontsize': 14}


def set_K_using_m(m):
    return np.ceil(np.exp(np.sqrt(np.log(2.)*np.log(m+1)))).astype(np.uint16)


def mplot(dumpfile, n, logeps, prior, edge, K, p, metric, **kwargs):
    out = np.empty(99)
    try:
        if type(K) == int:
            Ks, idx = [K], [0, 99]
        else:
            Ks, idx = np.unique(set_K_using_m(np.arange(1, 100)), return_index=True)
            idx = np.append(idx, 99)
        for K, a, b in zip(Ks, idx[:-1], idx[1:]):
            with h5py.File(dumpfile, 'r') as f:
                path = os.path.join(f'n={n}', f'logeps={logeps}', prior, edge, f'K={K}', f'p={p}', metric)
                out[a:b] = np.array(f[path])[:,a:b].mean(0)
    except:
        pdb.set_trace()
    plt.plot(np.arange(1, 100), out, **kwargs)


def static():

    if not os.path.isdir('static'):
        os.mkdir('static')

    for name in [
                 'Adult_age', 
                 'Adult_hours', 
                 'Goodreads_average_rating', 
                 'Goodreads_num_pages',
                 'Gaussian_0.0',
                 ]:
        for logeps in [0., -1.]:
            prior = 'Uniform'
            kwargs = {} if logeps else {'linestyle': 'dashed'}
            mplot(f'dump/{name}.h5', 1000, logeps, prior, 'cond', 2, 0., 'maxGap', label=None if kwargs else 'binary (AQ)', color='maroon', **kwargs)
            mplot(f'dump/{name}.h5', 1000, logeps, prior, 'edge', None, 1.5, 'maxGap', label=None if kwargs else 'K-ary (edge-based)', color='cornflowerblue', **kwargs)
            mplot(f'dump/{name}.h5', 1000, logeps, prior, 'cond', None, 1.5, 'maxGap', label=None if kwargs else 'K-ary (conditional)', color='darkorange', **kwargs)
        plt.xlabel('number of quantiles', **LEGEND)
        plt.ylabel(fr'maximum Gap ($\varepsilon=1.0$ and 0.1)', **LEGEND)
        plt.xticks(**LABELS)
        plt.yticks(**LABELS)
        plt.legend(**LEGEND, bbox_to_anchor=None if 'Gaussian' in name else (.4, .42))
        plt.tight_layout()
        plt.savefig(f'static/{name.replace("_0.0", "")}_mplot.png', dpi=256)
        plt.close()


def time_average(series, window):
    return np.convolve(series, np.ones(window), 'valid') / window

def timeplot(dumpfile, method, logeps, m, metric, window=30, **kwargs):
    path = os.path.join(method, f'logeps={logeps}', f'm={m}', metric)
    try:
        with h5py.File(dumpfile, 'r') as f:
            data = np.array(f[path]).mean(0)
    except: 
        pdb.set_trace()
    data = time_average(data, window)
    plt.plot(np.arange(len(data)), data, **kwargs)

def online_logepsplot(dumpfile, method, logepss, m, metric, agg=np.mean, **kwargs):
    out = []
    for logeps in logepss:
        path = os.path.join(method, f'logeps={logeps}', f'm={m}', metric)
        try:
            with h5py.File(dumpfile, 'r') as f:
                out.append(agg(np.array(f[path]), axis=1).mean())
        except:
            pdb.set_trace()
    plt.plot(logepss, out, **kwargs)


def online():

    if not os.path.isdir('online'):
        os.mkdir('online')

    m = 9
    for dataset, logepss, name in [
                                   ('CitiBike', np.append(np.linspace(-3., -1., 5), np.linspace(-.75, 0., 4)), 'citibike'),
                                   ('Synthetic_5.0', np.append(np.linspace(-2., -1., 3), np.linspace(-.75, 1., 8)), 'synthetic'),
                                   ('Flesch', np.append(np.linspace(-4., -1., 7), np.linspace(-.75, 1., 8)), 'worldnews'),
                                   ]:
        for agg in ['mean', 'median']:
            for method, kwargs in [
                                   ('Uniform', {'color': 'maroon', 'marker': 'o', 'label': 'Uniform (=AQ)'}),
                                   ('Cauchy', {'color': 'forestgreen', 'marker': 's'}),
                                   ('HalfCauchy', {'color': 'black', 'marker': 'x', 'label': 'half-Cauchy'}),
                                   ('PrevRelease-robust', {'color': 'darkorange', 'marker': '^', 'label': 'PubPrev (robust)'}),
                                   ('ProxyCOCOB-robust', {'color': 'cornflowerblue', 'marker': 'v', 'label': 'PubProx (robust)'}),
                                   ]:
                if method == 'HalfCauchy' and name != 'citibike':
                    continue
                kwargs['label'] = kwargs.get('label', method)
                online_logepsplot(f'dump/{dataset}.h5', method, logepss, m, 'avgGap', agg=getattr(np, agg), **kwargs)
            plt.xlabel(r'$\log_{10}(\varepsilon)$', **LEGEND)
            plt.ylabel(f'{agg} average Gap (m={m})', **LEGEND)
            plt.xticks(**LABELS)
            plt.yticks(**LABELS)
            plt.legend(**LEGEND)
            plt.tight_layout()
            plt.savefig(f'online/{name}_logepsplot_{agg}.png', dpi=256)
            plt.close()

    window = 1000
    for method, kwargs in [
                           ('Uniform', {'color': 'maroon', 'label': 'Uniform (=AQ)'}),
                           ('Cauchy', {'color': 'forestgreen'}),
                           ('DPFTRL-robust', {'color': 'darkorange', 'label': 'DP-FTRL (robust)'}),
                           ('ProxyCOCOB-robust', {'color': 'cornflowerblue', 'label': 'PubProx (robust)'}),
                           ('COCOB-robust', {'color': 'black', 'label': 'non-private OCO (robust)'}),
                           ]:
        kwargs['label'] = kwargs.get('label', method)
        timeplot('dump/Synthetic_0.0_T1E5.h5', method, -.5, 1, 'avgGap', window=window, **kwargs)
    plt.xlabel(f'time (window size {window})', **LEGEND)
    plt.ylabel('Gap (median)', **LEGEND)
    plt.xticks(**LABELS)
    plt.yticks(**LABELS)
    plt.legend(**LEGEND)
    plt.tight_layout()
    plt.savefig('online/synthetic_timeplot.png', dpi=256)
    plt.close()

    for method, kwargs in [
                           ('Uniform', {'color': 'maroon', 'label': 'Uniform (=AQ)'}),
                           ('PrevRelease', {'color': 'forestgreen', 'label': 'PubPrev'}),
                           ('PrevRelease-robust', {'color': 'darkorange', 'label': 'PubPrev (robust)'}),
                           ('ProxyCOCOB', {'color': 'black', 'label': 'PubProx'}),
                           ('ProxyCOCOB-robust', {'color': 'cornflowerblue', 'label': 'PubProx (robust)'}),
                           ]:
        kwargs['label'] = kwargs.get('label', method)
        timeplot('dump/CitiBike.h5', method, -2., m, 'avgGap', window=30, **kwargs)
    plt.xlabel('day (window size 30)', **LEGEND)
    plt.ylabel(f'average Gap (m={m})', **LEGEND)
    plt.xticks(**LABELS)
    plt.yticks(**LABELS)
    plt.legend(**LEGEND)
    plt.tight_layout()
    plt.savefig('online/citibike_timeplot.png', dpi=256)
    plt.close()


def pubpri_logepsplot(dumpfile, source, key, m, n, logepss, method, metric, flat=False, **kwargs):
    out = []
    for logeps in logepss:
        path = os.path.join(f'{source}-{key}-n{n}-m{m}', f'logeps={logeps}', method, metric)
        try:
            with h5py.File(dumpfile, 'r') as f:
                out.append(np.array(f[path]).mean(0))
        except:
            pdb.set_trace()
        if metric == 'negLogPsi':
            out[-1] *= m / 10. ** logeps
    if flat:
        out = np.array(out).mean() * np.ones(len(out))
    plt.plot(logepss, out, **kwargs)

def mixplot(dumpfile, source, key, m, n, logeps, method, metric, **kwargs):
    path = os.path.join(f'{source}-{key}-n{n}-m{m}', f'logeps={logeps}', method, metric)
    try:
        with h5py.File(dumpfile, 'r') as f:
            out = np.array(f[path]).mean(0)
    except: 
        pdb.set_trace()
    plt.plot(np.linspace(0., 1., len(out)), out, **kwargs)

def pubpri():

    if not os.path.isdir('pubpri'):
        os.mkdir('pubpri')

    m = 9
    for key in ['age', 'hours']:
        for method, kwargs in [
                               ('Uniform', {'color': 'maroon', 'marker': 'o', 'label': 'Uniform (=AQ)'}),
                               ('public-quantiles', {'color': 'black', 'flat': True, 'linestyle': 'dashed', 'label': 'public quantiles'}),
                               ('public-Cauchy', {'color': 'forestgreen', 'marker': 's', 'label': 'public Cauchy'}),
                               ('DEFit', {'color': 'darkorange', 'marker': '^', 'label': 'PubFit'}),
                               ('DEFit-robust', {'color': 'cornflowerblue', 'marker': 'v', 'label': 'PubFit (robust)'}),
                               ]:
            kwargs['label'] = kwargs.get('label', method)
            pubpri_logepsplot('dump/test-pubpri_Non.h5', 'train', key, m, 100, np.linspace(-3., 1., 9), method, 'avgGap', **kwargs)
        plt.xlabel(r'$\log_{10}(\varepsilon)$', **LEGEND)
        plt.ylabel(f'average Gap (m={m})', **LEGEND)
        plt.xticks(**LABELS)
        plt.yticks(**LABELS)
        plt.legend(**LEGEND)
        plt.tight_layout()
        plt.savefig(os.path.join('pubpri', f'{key}_logepsplot.png'), dpi=256)
        plt.close()

    for key in ['average_rating', 'num_pages']:
        for method, kwargs in [
                               ('Uniform', {'color': 'maroon', 'marker': 'o', 'label': 'Uniform (=AQ)'}),
                               ('public-quantiles', {'color': 'black', 'linestyle': 'dashed', 'label': 'public quantiles'}),
                               ('public-Cauchy', {'color': 'forestgreen', 'marker': 's', 'label': 'public Cauchy'}),
                               ('DEFit', {'color': 'darkorange', 'marker': '^', 'label': 'PubFit'}),
                               ('DEFit-robust', {'color': 'cornflowerblue', 'marker': 'v', 'label': 'PubFit (robust)'}),
                               ]:
            kwargs['label'] = kwargs.get('label', method)
            mixplot('dump/poetry-pubpri_Non.h5', 'history', key, m, 100, 0., method, 'avgGap', **kwargs)
        plt.xlabel('fraction of samples coming from "Poetry" genre', **LEGEND)
        plt.ylabel(f'average Gap (m={m})', **LEGEND)
        plt.xticks(**LABELS)
        plt.yticks(**LABELS)
        plt.legend(**LEGEND)
        plt.tight_layout()
        plt.savefig(os.path.join('pubpri', f'{key}_mixplot.png'), dpi=256)
        plt.close()


if __name__ == '__main__':

    static()
    online()
    pubpri()
