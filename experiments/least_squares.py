#%%

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace

from virn.logging import Metrics
from virn.methods import OG, DualAvg, DualX, DualOpt
from virn.problems import KellyAuction, KellyAuction2D, KellyAuction4D, NoiseProfile
from virn.plot import plot_kelly, savefigs

#%% Least square (const. vs adaptive)
metrics_dict = {}
configs = [
    Namespace(adaptive=True, lr=1.0),
    Namespace(adaptive=False, lr=0.01),
]

for config in configs:
    kelly = KellyAuction4D()
    kelly.equilibrium = np.zeros(kelly.dim())
    noise_profile = NoiseProfile(sigma_abs=1.0, sigma_rel=0.0)

    T = 100000
    N  = 1
    X = 100 * np.ones((N, kelly.dim()))
    method = DualX(lr=config.lr, adaptive=config.adaptive)
    metrics = Metrics(T)

    for t in range(T):
        grad = X#kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'adapt={config.adaptive}'] = metrics

figs, axes = plot_kelly(kelly, metrics_dict, T, num_points=1000)
#savefigs("dim4_noise_compare", figs)
