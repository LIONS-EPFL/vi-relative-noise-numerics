import numpy as np


class Metrics(object):
    def __init__(self, T) -> None:
        self.T = T
        self.metrics = {}

    def add(self, name, ndarray, t):
        if name in self.metrics:
            self.metrics[name][t] = ndarray
        else:
            self.metrics[name] = np.zeros((self.T,) + ndarray.shape)
            self.metrics[name][t] = ndarray

    def __getitem__(self, key):
        return self.metrics[key]
