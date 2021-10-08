# GD Training on the expected loss with weight regularization
#%%

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from virn.methods import DualX, GDA, OG, DualOpt
from virn.problems import CovarianceLearning
from virn.plot import plot_covariance_weights

# Parameters of distribution
dim = 2
np.random.seed(27)
U = np.random.normal(size=(dim,dim)) + 0.1
print("U = {}".format(U))
Sigma = np.dot(U, U.T)
print("Sigma = {}".format(Sigma))
x = np.random.multivariate_normal(mean = np.zeros(dim),
                                  cov = Sigma, size=1000)
if dim == 2:
    plt.scatter(x[:,0], x[:,1])
if dim == 1:
    plt.hist(x)
plt.show()

problem = CovarianceLearning(Sigma, dim=dim)
#%% GDA
T = 10000
clip_D = 1
lr = 0.02
lambda_reg = .1
W = 0.1*np.ones((T, dim, dim))
V = 0.2*np.ones((T, dim, dim))
minibatch = 50

method = GDA(lr=lr)

nabla_w = np.zeros((T, dim, dim))
nabla_v = np.zeros((T, dim, dim))

for t in range(2,T):
    grads = problem.grad(
        V[t-1], W[t-1], 
        minibatch=minibatch, 
        lambda_reg=lambda_reg)
    nabla_v[t-1],nabla_w[t-1] = grads
    grad = np.stack(grads, axis=-1)
    Z = np.stack([V[t-1], W[t-1]], axis=-1)
    Z = method.step(Z, grad)
    V[t], W[t] = [Z[...,0], Z[...,1]]

plot_covariance_weights(W, V, Sigma, T, save_prefix='gd')

#%% OG
T = 10000
clip_D = 1
lr = 0.02
lambda_reg = .1
W = 0.1*np.ones((T, dim, dim))
V = 0.2*np.ones((T, dim, dim))
minibatch = 50

method = OG(lr=lr)

nabla_w = np.zeros((T, dim, dim))
nabla_v = np.zeros((T, dim, dim))

for t in range(2,T):
    grads = problem.grad(
        V[t-1], W[t-1], 
        minibatch=minibatch, 
        lambda_reg=lambda_reg)
    nabla_v[t-1],nabla_w[t-1] = grads
    grad = np.stack(grads, axis=-1)
    Z = np.stack([V[t-1], W[t-1]], axis=-1)
    Z = method.step(Z, grad)
    V[t], W[t] = [Z[...,0], Z[...,1]]

plot_covariance_weights(W, V, Sigma, T, save_prefix='og')

#%% Adaptive DualX
T = 10000
clip_D = 1
lr = 1.0
lambda_reg = .1
W = 0.1*np.ones((T, dim, dim))
V = 0.2*np.ones((T, dim, dim))
minibatch = 50

method = DualX(lr=lr)
#method = DualOpt(lr=lr)
#method = DualAvg(lr=lr)

nabla_w = np.zeros((T, dim, dim))
nabla_v = np.zeros((T, dim, dim))

for t in range(2,T):
    grads = problem.grad(
        V[t-1], W[t-1], 
        minibatch=minibatch, 
        lambda_reg=lambda_reg)
    nabla_v[t-1],nabla_w[t-1] = grads
    grad = np.stack(grads, axis=-1)
    Z = np.stack([V[t-1], W[t-1]], axis=-1)
    Z = method.step(Z, grad)
    V[t], W[t] = [Z[...,0], Z[...,1]]

plot_covariance_weights(W, V, Sigma, T, save_prefix='oda')

# %%
