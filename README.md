# VIs with relative noise


## Setup

```
python setup.py develop
```

## File structure

Experiments (they support vscode's matlab like block execution):

```
├── experiments
│   ├── kelly_auction2d.py
│   ├── kelly_auction.py
│   ├── kelly_auction_highdim.py
│   ├── learning_covariance.py
│   └── least_squares.py
```

Code base:

```
├── virn
│   ├── logging.py
│   ├── methods.py
│   ├── plot.py
│   └── problems.py
  ```

Optimal solution for the Kelly auction is computed with the Mathematica script found in [`Kelly.nb`](Kelly.nb).


## Usage

See [`experiments/`](experiments/) to recreate existing experiments.
They can be run interactively in e.g. with vscodes block execution or directly from the terminal:

```
cd experiments
python kelly_action.py
python kelly_action_highdim.py
python kelly_action_covariance.py
```

To try a different configuration see the example script below:

```python
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace

from virn.logging import Metrics
from virn.methods import OG, DualAvg, DualX, DualOpt
from virn.problems import KellyAuction, KellyAuction2D, KellyAuction4D, NoiseProfile
from virn.plot import plot_kelly, savefigs


# Use Namespace instead of a dict for easy dot notation access (e.g. config.name)
metrics_dict = {}
configs = [
    Namespace(name="DualX non-adaptive", method=DualX(lr=1.0, adaptive=False)),
    Namespace(name="DualX", method=DualX(lr=200.0)),
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

# Generate and save the plots
figs, axes = plot_kelly(kelly, metrics_dict, T, num_points=1000)
savefigs("example", figs)
```

## Preparation

> **Note**: This was only a problem prior to introducing the `num_points` limit in `plot_kelly`.

To avoid humongous files in the writeup we can convert them to png (in case they were not generated):

```
find . -maxdepth 1 -type f -name '*.pdf' -exec pdftoppm -r 300 -png {} {} \;
```

This requires poppler on macOS which can be install with `brew install poppler`.
