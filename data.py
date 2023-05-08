import gzip
import json
import os
import pdb
import pickle
import requests
import sys
import time
import zipfile
from collections import namedtuple
from datetime import datetime
from operator import itemgetter
import gdown
import nltk
import numpy as np
import textstat
import torch
from mpi4py import MPI
from nltk.corpus import stopwords


Dataset = namedtuple('Dataset', ['name', 'data', 'features', 'guesses'])
Guesses = namedtuple('Guesses', ['interval', 'loc', 'scale', 'qbound', 'nonnegative'])
cheat = {}


def synthetic_features(T=2500, n_features=10, n_quantiles=9, rate=0., min_count=100, **kwargs):

    X, F = [], []
    w, b = np.random.normal(0., 1., n_features), np.sort(np.random.normal(0., 1., n_quantiles+2))
    cheat['w'] = w
    cheat['b'] = b
    for t in range(T):
        f = np.random.normal(0., 1., n_features)
        bounds = np.inner(w, f) + b
        X.append(np.sort(np.hstack([np.random.uniform(a, b, int(np.ceil(min_count/(n_quantiles+1)))+np.random.poisson(rate))
                                    for a, b in zip(bounds[:-1], bounds[1:])])))
        F.append(torch.Tensor(f))
    return X, F

Synthetic = Dataset('Synthetic', None, synthetic_features, Guesses(1E-8, 0., 1., lambda q: (-100., 100.), False))


def citibike_features(min_trips=10, days=True, weather=True, verbose=True, **kwargs):

    root = 'data/citibike'
    if not os.path.isdir(root):
        os.mkdir(root)

    pkl = os.path.join(root, 'citibike.pkl')
    if os.path.isfile(pkl) or MPI.COMM_WORLD.rank:
        while True:
            try:
                with open(pkl, 'rb') as f:
                    citibike = pickle.load(f)
                break
            except FileNotFoundError:
                time.sleep(60)

    else:
        if verbose:
            print('loading ride data')
            sys.stdout.flush()
        url = "https://s3.amazonaws.com/tripdata/"
        start = '2015-09'
        end = '2022-12'
        citibike = {day: [] for day in np.arange(np.datetime64(start+'-01'), np.datetime64(end+'-01'))}

        for month in np.arange(np.datetime64(start), np.datetime64(end)):
            month = str(month)
            if verbose:
                print('\r', month, end=' ')
                sys.stdout.flush()
            zfname = os.path.join(root, month + '.csv.zip')
            fname = 'JC-' + month.replace('-', '')
            if month == '2017-08':
                fname += ' '
            else:
                fname += '-'
            if month == '2022-07':
                fname += 'citbike-tripdata.csv'
            else:
                fname += 'citibike-tripdata.csv'
            if not os.path.isfile(zfname):
                response = requests.get(url + fname + '.zip')
                with open(zfname, 'wb') as f:
                    f.write(response.content)

            datadir = zfname.split('.')[0]
            if not os.path.isdir(datadir):
                os.mkdir(datadir)
                with zipfile.ZipFile(zfname, 'r') as f:
                    f.extractall(datadir)

            if month in {'2016-01', '2016-02', '2016-03'}:
                fname = fname[:7] + fname[8:]
            with open(os.path.join(datadir, fname), 'r') as f:
                line = f.readline()
                for line in f:
                    if month < '2021-02':
                        length, date = line.split(',')[:2]
                        length = int(length)
                        date = np.datetime64(date[:date.index(' ')].strip('"'))
                    else:
                        date, end = line.split(',')[2:4]
                        length = (np.datetime64(end.strip('"')) - np.datetime64(date.strip('"'))).astype('int')
                        date = np.datetime64(date[:date.index(' ')].strip('"'))
                    citibike[date].append(length)

        with open(pkl, 'wb') as f:
            pickle.dump(citibike, f)
        if verbose:
            print('\rloading feature data')
            sys.stdout.flush()

    X, F = [], []
    with open(os.path.join(root, 'centralpark.csv'), 'r') as f:
        f.readline()
        for i, (date, lengths) in enumerate(sorted(citibike.items(), key=itemgetter(0))):
            weekday = [int(j == i % 7) for j in range(7)] if days else []
            dt = date.item()
            yearday = [np.sin(2. * np.pi * dt.timetuple().tm_yday / (365. + int(dt.year % 4 == 0)))] if days else []
            weather = [entry.replace('"', '').strip() for entry in f.readline().split(',')]
            cond = [float(weather[j]) / 10. if weather[j] else 0. for j in [4, 6, 7, 8]] if weather else []
            temp = [float(weather[j]) / 100. for j in [10, 11]] if weather else []
            if len(lengths) > min_trips:
                X.append(np.sort(np.array(lengths)) / 60.)
                F.append(torch.Tensor(weekday + yearday + temp + cond) if days or weather else torch.zeros(1))

    return X, F

CitiBike = Dataset('CitiBike', None, citibike_features, Guesses(1./60., 10., 1., lambda q: (0., 50./(1.-q)), True))


def goodreads_data(verbose=True):

    root = 'data/goodreads'
    if not os.path.isdir(root):
        os.mkdir(root)

    pkl = os.path.join(root, 'goodreads.pkl')
    if os.path.isfile(pkl) or MPI.COMM_WORLD.rank:
        while True:
            try:
                with open(pkl, 'rb') as f:
                    goodreads = pickle.load(f)
                break
            except FileNotFoundError:
                time.sleep(60)

    else:
        goodreads = {}
        keys = ['average_rating', 'num_pages']
        for genre, url in [
                           ('children', 'https://drive.google.com/uc?id=1R3wJPgyzEX9w6EI8_LmqLbpY4cIC9gw4'),
                           ('comics', 'https://drive.google.com/uc?id=1ICk5x0HXvXDp5Zt54CKPh5qz1HyUIn9m'),
                           ('fantasy', 'https://drive.google.com/uc?id=1x8IudloezYEg6qDTPxuBkqGuQ3xIBKrt'),
                           ('history', 'https://drive.google.com/uc?id=1roQnVtWxVE1tbiXyabrotdZyUY7FA82W'),
                           ('mystery', 'https://drive.google.com/uc?id=1ACGrQS0sX4-26D358G2i5pja1Y6CsGtz'),
                           ('poetry', 'https://drive.google.com/uc?id=1H6xUV48D5sa2uSF_BusW-IBJ7PCQZTS1'),
                           ('romance', 'https://drive.google.com/uc?id=1juZreOlU4FhGnBfP781jAvYdv-UPSf6Q'),
                           ('young_adult', 'https://drive.google.com/uc?id=1gH7dG4yQzZykTpbHYsrw2nFknjUm0Mol'),
                           ]:
            fname= os.path.join(root, f'{genre}.json.gz')
            if not os.path.isfile(fname):
                gdown.download(url, fname, quiet=not verbose)
            data = {key: [] for key in keys}
            if verbose:
                print('loading', genre, 'data')
                sys.stdout.flush()
            with gzip.open(fname, 'rb') as f:
                for line in f:
                    entry = json.loads(line)
                    for key in keys:
                        if entry[key]:
                            data[key].append(float(entry[key]))
            goodreads[genre] = {key: np.sort(data[key]) for key in keys}

        with open(pkl, 'wb') as f:
            pickle.dump(goodreads, f)

    return goodreads

Goodreads = Dataset('Goodreads', goodreads_data, None, {'average_rating': Guesses(.01, 2.5, .5, lambda q: (0., 5.), True),
                                                        'num_pages': Guesses(1., 200., 25., lambda q: (0., 1000./(1.-q)), True)})


def adult_data(verbose=True):

    root = 'data/adult'
    if not os.path.isdir(root):
        os.mkdir(root)

    pkl = os.path.join(root, 'adult.pkl')
    if os.path.isfile(pkl) or MPI.COMM_WORLD.rank:
        while True:
            try:
                with open(pkl, 'rb') as f:
                    adult = pickle.load(f)
                break
            except FileNotFoundError:
                time.sleep(60)

    else:
        adult = {}
        for subset, url in [
                            ('train', 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'),
                            ('test', 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'),
                             ]:
            fname = os.path.join(root, f'{subset}.csv')
            if not os.path.isfile(fname):
                response = requests.get(url)
                with open(fname, 'wb') as f:
                    f.write(response.content)
            data = {'age': [], 'hours': []}
            with open(fname, 'r') as f:
                if subset == 'test':
                    f.readline()
                for line in f:
                    line = line.split(',')
                    if len(line) == 1:
                        break
                    data['age'].append(float(line[0]))
                    data['hours'].append(float(line[-3]))
            adult[subset] = {key: np.sort(value) for key, value in data.items()}

        with open(pkl, 'wb') as f:
            pickle.dump(adult, f)

    return adult

Adult = Dataset('Adult', adult_data, None, {'age': Guesses(1., 40., 5., lambda q: (10., 120.), True),
                                            'hours': Guesses(1., 40., 2., lambda q: (0., 168.), True)})

Gaussian = Dataset('Gaussian',
                   lambda: {'0.0': {None: np.random.normal(0., 1., 1000000)},
                            '2.0': {None: np.random.normal(2., 1., 1000000)},
                            '4.0': {None: np.random.normal(4., 1., 1000000)}},
                   None,
                   {None: Guesses(1E-8, 0., 5., lambda q: (-10., 10.), False)})

def worldnews(target='flesch', min_comments=10, verbose=False, **kwargs):

    root = 'data/worldnews'
    commentfile = os.path.join(root, 'comments.pkl')
    postfile = os.path.join(root, 'posts.pkl')
    if not os.path.isfile(commentfile) or not os.path.isfile(postfile):

        if verbose:
            print('loading conversations')
            sys.stdout.flush()
        with open(os.path.join(root, 'bbc.pkl'), 'rb') as f:
            bbc = pickle.load(f)
        with open(os.path.join(root, 'conversations.json'), 'r') as f:
            convs = json.load(f)

        comments, posts = {}, {}
        for i, value in enumerate(bbc.values()):
            if value['text']:
                conv = value['root']
                if not conv in comments:
                    comments[conv] = {'flesch': [], 'score': [], 'time': [], 'tokens': []}
                    posts[conv] = {'timestamp': datetime.fromtimestamp(convs[conv]['timestamp']),
                                   'flesch':  textstat.flesch_reading_ease(convs[conv]['title']),
                                   'title': convs[conv]['title'],
                                   'gilded': convs[conv]['gilded'] != 0}
                comments[conv]['flesch'].append(textstat.flesch_reading_ease(value['text']))
                comments[conv]['score'].append(value['meta']['score'])
                comments[conv]['time'].append(value['timestamp'] - convs[conv]['timestamp'])
                comments[conv]['tokens'].append(len(value['text'].split()))
            if verbose and not (i+1) % 1000:
                print('loaded', i+1, 'comments', end='\r')
                sys.stdout.flush()

        with open(commentfile, 'wb') as f:
            pickle.dump(comments, f)
        with open(postfile, 'wb') as f:
            pickle.dump(posts, f)

    else:
        with open(commentfile, 'rb') as f:
            comments = pickle.load(f)
        with open(postfile, 'rb') as f:
            posts = pickle.load(f)

    embeddingfile = os.path.join(root, 'embeddings.pkl')
    if os.path.isfile(embeddingfile):
        with open(embeddingfile, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        if verbose:
            print('loading word embeddings')
            sys.stdout.flush()
        wv = {}
        with open(os.path.join(root, 'glove.twitter.27B.25d.txt'), 'r') as f:
            for line in f:
                idx = line.index(' ')
                wv[line[:idx]] = np.fromstring(line[idx:], sep=' ')
        embeddings = {}
        nltk.download('stopwords')
        stop = stopwords.words('english')
        for i, (post, value) in enumerate(posts.items()):
            embeddings[post] = np.zeros(25)
            for token in value['title'].lower().split():
                if token in wv and not token in stop:
                    embeddings[post] += wv[token]
            if verbose and not (i+1) % 1000:
                print('build', i+1, 'embeddings', end='\r')
        with open(embeddingfile, 'wb') as f:
            pickle.dump(embeddings, f)
    
    keytime = []
    for post, value in posts.items():
        zeros = sum(t == 0 for t in comments[post]['time'])
        if len(comments[post][target]) - zeros > min_comments:
            keytime.append((post, value['timestamp']))
        comments[post][target] = comments[post][target][zeros:]
    keytime = sorted(keytime, key=itemgetter(1))
    X, F = [], []
    for key, dt in keytime:
        x = np.sort(comments[key][target])
        X.append(x)
        features = [0 for _ in range(7)]
        features[datetime.weekday(dt)] = 1.
        tt = dt.timetuple()
        features.append(np.sin(2. * np.pi * tt.tm_yday / (365. + int(dt.year % 4 == 0))))
        features.append(np.sin(2. * np.pi * (tt.tm_hour+tt.tm_min/60.+tt.tm_sec/3600.) / 24.))
        post = posts[key]
        features.append(int(post['gilded']))
        features.append(post['flesch'] / 100.)
        features.append(len(post['title'].split()) / 100.)
        embedding = embeddings[key]
        norm = np.linalg.norm(embedding)
        if norm:
            embedding /= norm
        F.append(torch.Tensor(np.append(features, embedding)))

    return X, F
        
Flesch = Dataset('Flesch', None, worldnews, Guesses(.01, 50., 10., lambda q: (-100/q-100, 100+q*100), False))
