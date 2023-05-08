import pdb
import sys
import time
import h5py
import numpy as np
from mpi4py import MPI


def write(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def dump(metrics, fname, path, verbose=True):
    comm = MPI.COMM_WORLD
    if comm.rank:
        for sendbuf in metrics:
            comm.Gatherv(sendbuf, None, root=0)
    else:
        if verbose:
            write('aggregating', end='\r')
        for i, (metric, sendbuf) in enumerate(metrics._asdict().items()):
            recvbuf = np.empty((comm.size, len(sendbuf)))
            comm.Gatherv(sendbuf, recvbuf, root=0)
            while True:
                j = 0
                try:
                    with h5py.File(fname, 'a') as f:
                        g = f[path] if i else f.create_group(path)
                        g.create_dataset(metric, data=recvbuf)
                    break
                except OSError:
                    time.sleep(10)
                    j += 1
                    if verbose:
                        write(f'open attempt ({j})', end='\r')
        if verbose:
            write('completed', fname, path)
