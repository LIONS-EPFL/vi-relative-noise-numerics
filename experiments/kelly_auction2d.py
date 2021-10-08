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
from virn.plot import plot_kelly

#kelly = KellyAuction(G=np.array([2.0, 3.0]), Q=1.0, Z=1.0)
kelly = KellyAuction2D()
noise_profile = NoiseProfile(sigma_abs=0.0, sigma_rel=0.2)

#%% 

metrics_dict = {}
configs = [
    Namespace(sigma_abs=0.0, sigma_rel=0.0),
    Namespace(sigma_abs=0.1, sigma_rel=0.0),
    Namespace(sigma_abs=0.0, sigma_rel=0.1),
]

for config in configs:
    kelly = KellyAuction2D()
    noise_profile = NoiseProfile(sigma_abs=config.sigma_abs, sigma_rel=config.sigma_rel)

    T = 10000
    X = np.zeros((1, kelly.dim()))
    method = DualX()
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'$\\sigma_{{\\mathrm{{abs}}}}={config.sigma_abs},\\sigma_{{\\mathrm{{rel}}}}={config.sigma_rel}$'] = metrics

figs, axes = plot_kelly(kelly, metrics_dict, T)
metrics['trajectory'][-1]
#%%

# %%

metrics_dict = {}
configs = [
    Namespace(sigma_abs=0.01, sigma_rel=0.0),
    Namespace(sigma_abs=0.1, sigma_rel=0.0),
]

for config in configs:
    kelly = KellyAuction2D()
    noise_profile = NoiseProfile(sigma_abs=config.sigma_abs, sigma_rel=config.sigma_rel)
    
    T = 10000
    X = np.zeros((1, kelly.dim()))
    method = DualX()
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'$\\sigma_{{\\mathrm{{abs}}}}={config.sigma_abs}$'] = metrics

figs, axes = plot_kelly(kelly, metrics_dict, T)

#%%

metrics_dict = {}
configs = [
    Namespace(sigma_abs=0.0, sigma_rel=0.0),
    Namespace(sigma_abs=0.0, sigma_rel=0.1),
    Namespace(sigma_abs=0.0, sigma_rel=1.0),
    Namespace(sigma_abs=0.0, sigma_rel=10.0),
]

for config in configs:
    kelly = KellyAuction2D()
    noise_profile = NoiseProfile(sigma_abs=config.sigma_abs, sigma_rel=config.sigma_rel)
    
    T = 10000
    X = np.zeros((1, kelly.dim()))
    method = DualX()
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'$\\sigma_{{\\mathrm{{rel}}}}={config.sigma_rel}$'] = metrics

figs, axes = plot_kelly(kelly, metrics_dict, T)
