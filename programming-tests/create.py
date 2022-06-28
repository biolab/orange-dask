import h5py
import dask.array as da
import numpy as np


def save_dask(x, y=None, filename=""):
    with h5py.File(filename, 'w') as f:
        f.create_dataset("X", data=x)
        if y is not None:
            f.create_dataset("Y", data=y)


def t4e6_raw(fn):
    rand = np.random.RandomState(0)
    x = np.c_[rand.random((20000, 10000)), rand.randint(4, size=(20000, 10000))]
    save_dask(x, filename=fn)


def t4e6_raw_y(fn):
    rand = np.random.RandomState(0)
    x = np.c_[rand.random((20000, 10000)), rand.randint(4, size=(20000, 10000))]
    y = np.c_[rand.random((20000, 10000)), rand.randint(4, size=(20000, 10000))]
    save_dask(x, y=y, filename=fn)


def read_raw(filename):
    f = h5py.File(filename, "r")
    X = f['X']
    X = da.from_array(X)
    if 'Y' in f:
        Y = f['Y']
        Y = da.from_array(Y)
        return X, Y
    return X
