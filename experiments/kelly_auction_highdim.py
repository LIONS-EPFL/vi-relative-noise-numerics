#%%

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
import math

from virn.logging import Metrics
from virn.methods import OG, DualAvg, DualX, DualOpt
from virn.problems import KellyAuction, KellyAuction2D, KellyAuction4D, NoiseProfile
from virn.plot import plot_kelly, savefigs


#%%
if False:
    sol_metrics_dict = {}
    configs = [
        Namespace(name="optimal", method=DualX(lr=100)),
    ]


    for config in configs:
        num_players = 100
        kelly = KellyAuction(G=6+np.linspace(0, 0.1, num_players), Q=1000.0, Z=100.0)
        kelly.equilibrium = 200 * np.ones(num_players)
        noise_profile = NoiseProfile(sigma_abs=0.0, sigma_rel=0.0)

        T = 5000000
        interval = 100000
        num_points = math.ceil(T / interval)
        N  = 1
        X = np.zeros((N, kelly.dim()))
        metrics = Metrics(num_points)

        i = 0
        for t in range(T):
            grad = kelly.grad(X)
            stoc_grad = noise_profile.add_noise(grad)
            X = config.method.step(X, stoc_grad)
            if t % interval == 0:
                metrics.add('trajectory', X, i)
                metrics.add('grads', grad, i)
                metrics.add('stoc_grads', stoc_grad, i)
                i += 1
                print(f"iteration {t}")

        sol_metrics_dict[config.name] = metrics

    figs, axes = plot_kelly(kelly, sol_metrics_dict, num_points)
    eq = sol_metrics_dict['optimal']['trajectory'][-1]
else:
    eq = np.load("kelly_auction_100d_equilibrium.npy")
#%% Uncomment `save` to update the current numerically found equilibrium
#eq = sol_metrics_dict['optimal']['trajectory'][-1]
#np.save("kelly_auction_100d_equilibrium.npy", eq)
eq
#%%
# kelly.equilibrium = 100 * np.ones(num_players)
# figs, axes = plot_kelly(kelly, sol_metrics_dict, num_points)

# %% Importance of diameter

T = 1000000
num_players = 100
kelly = KellyAuction(G=6+np.linspace(0, 0.1, num_players), Q=1000.0, Z=100.0)
kelly.equilibrium = eq

#%%
metrics_dict = {}
configs = [
    Namespace(name="D=100", method=DualX(lr=100.0)),
    # Namespace(name="D=10", method=DualX(lr=10.0)),
    # Namespace(name="D=5", method=DualX(lr=5.0)),
    # Namespace(name="D=1", method=DualX(lr=1)),
]

interval = 1
num_points = math.ceil(T / interval)

for config in configs:
    noise_profile = NoiseProfile(sigma_abs=0.0, sigma_rel=0.1)

    N  = 10
    X = np.zeros((N, kelly.dim()))
    metrics = Metrics(num_points)

    i = 0
    for t in range(T):
        grad = kelly.grad(X)
        stoc_grad = noise_profile.add_noise(grad)
        X = config.method.step(X, stoc_grad)

        if t % interval == 0:
            metrics.add('trajectory', X, i)
            metrics.add('grads', grad, i)
            metrics.add('stoc_grads', stoc_grad, i)
            i += 1

    metrics_dict[config.name] = metrics

#%%

figs, axes = plot_kelly(kelly, metrics_dict, num_points, num_points=1000)

# %%

savefigs("highdim_compare", figs)
# %%
