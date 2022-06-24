import h5py
import dask.array as da
import numpy as np


def save_dask(x, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset("X", data=x)


def t4e6_raw(fn):
    x = np.c_[np.random.random((20000, 10000)), np.random.randint(4, size=(20000, 10000))]
    save_dask(x, fn)


def read_raw(filename):
    f = h5py.File(filename, "r")
    X = f['X']
    X = da.from_array(X)
    return X
