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


# %%

metrics_dict = {}
configs = [
    Namespace(name="OG", method=OG(lr=1.0)),
    Namespace(name="DualX non-adapt", method=DualX(lr=2.0, adaptive=False)),
    Namespace(name="DualX", method=DualX(lr=200.0)),
    Namespace(name="DualAvg", method=DualAvg(lr=200.0)),
    Namespace(name="DualOpt", method=DualOpt(lr=200.0)),
]

for config in configs:
    kelly = KellyAuction4D()
    noise_profile = NoiseProfile(sigma_abs=0.0, sigma_rel=0.1)
    
    T = 10000
    N  = 10
    X = np.zeros((N, kelly.dim()))
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = config.method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[config.name] = metrics

figs, axes = plot_kelly(kelly, metrics_dict, T, num_points=1000)
savefigs("dim4_method_compare", figs)

# %%

metrics_dict = {}
configs = [
    Namespace(sigma_abs=0.0, sigma_rel=0.0),
    Namespace(sigma_abs=1.0, sigma_rel=0.0),
    Namespace(sigma_abs=0.0, sigma_rel=1.0),
]

for config in configs:
    kelly = KellyAuction4D()
    noise_profile = NoiseProfile(sigma_abs=config.sigma_abs, sigma_rel=config.sigma_rel)

    T = 1000000
    N  = 10
    X = np.zeros((N, kelly.dim()))
    method = DualX(lr=100.0)
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'$\\sigma_{{\\mathrm{{abs}}}}={config.sigma_abs},\\sigma_{{\\mathrm{{rel}}}}={config.sigma_rel}$'] = metrics

#%%
figs, axes = plot_kelly(kelly, metrics_dict, T,  ylim_factor=[0.1, 10.0], num_points=1000)
savefigs("dim4_noise_v2_compare", figs)

#%%

metrics_dict = {}
configs = [
    Namespace(sigma_abs=0.0, sigma_rel=0.0),
    Namespace(sigma_abs=0.0, sigma_rel=0.1),
    Namespace(sigma_abs=0.0, sigma_rel=1.0),
    Namespace(sigma_abs=0.0, sigma_rel=10.0),
]

for config in configs:
    kelly = KellyAuction4D()
    noise_profile = NoiseProfile(sigma_abs=config.sigma_abs, sigma_rel=config.sigma_rel)
    
    T = 1000000
    N  = 10
    X = np.zeros((N, kelly.dim()))
    method = DualX(lr=100.0)
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'$\\sigma_{{\\mathrm{{rel}}}}={config.sigma_rel}$'] = metrics

# figsize=[6, 4]
figs, axes = plot_kelly(kelly, metrics_dict, T, num_points=1000)
savefigs("dim4_rel_noise_compare", figs)

# %%

metrics_dict = {}
configs = [
    Namespace(sigma_abs=0.01, sigma_rel=0.0),
    Namespace(sigma_abs=0.1, sigma_rel=0.0),
    Namespace(sigma_abs=1.0, sigma_rel=0.0),
]

for config in configs:
    kelly = KellyAuction4D()
    noise_profile = NoiseProfile(sigma_abs=config.sigma_abs, sigma_rel=config.sigma_rel)
    
    T = 1000000
    N  = 10
    X = np.zeros((N, kelly.dim()))
    method = DualX(lr=100.0)
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'$\\sigma_{{\\mathrm{{abs}}}}={config.sigma_abs}$'] = metrics

# figsize=[6, 4]
figs, axes = plot_kelly(kelly, metrics_dict, T, num_points=1000)
savefigs("dim4_abs_noise_compare", figs)

# %%

metrics_dict = {}
configs = [
    #Namespace(sigma_abs=0.0, sigma_rel=0.0),
    Namespace(sigma_abs=0.0, sigma_rel=0.1),
    Namespace(sigma_abs=0.0, sigma_rel=1.0),
    #Namespace(sigma_abs=0.0, sigma_rel=10.0),
]

for config in configs:
    kelly = KellyAuction4D()
    noise_profile = NoiseProfile(sigma_abs=config.sigma_abs, sigma_rel=config.sigma_rel)
    
    T = 1000000
    N  = 10
    X = np.zeros((N, kelly.dim()))
    method = DualX(lr=100.0)
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'$\\sigma_{{\\mathrm{{rel}}}}={config.sigma_rel}$'] = metrics

# figsize=[6, 4]
figs, axes = plot_kelly(kelly, metrics_dict, T, num_points=1000)
savefigs("dim4_rel_noise_compare_few", figs)
#%%
figs.average_iter

# %%


metrics_dict = {}
configs = [
    # Namespace(D=1000.0),
    Namespace(D=500.0),
    Namespace(D=200.0),
    Namespace(D=50.0),
    Namespace(D=10.0),
]

for config in configs:
    kelly = KellyAuction4D()
    noise_profile = NoiseProfile(sigma_abs=0.0, sigma_rel=0.2)

    T = 100000
    N  = 10
    X = np.zeros((N, kelly.dim()))
    method = DualX(lr=config.D)
    metrics = Metrics(T)

    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = method.step(X, stoc_grad)
        metrics.add('trajectory', X, t)
        metrics.add('grads', grad, t)
        metrics.add('stoc_grads', stoc_grad, t)

    metrics_dict[f'$D={config.D}$'] = metrics
#%%

figs, axes = plot_kelly(kelly, metrics_dict, T, num_points=1000)
savefigs("dim4_D_compare", figs)

# %%
