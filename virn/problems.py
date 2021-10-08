from os import initgroups
import numpy as np
import matplotlib.pyplot as plt


class Problem(object):
    def plot_vectorfield(self, ax=None, N=15, M=15, show_legend=False, bounds=None):
        """Plot the gradient vector field of minmax dynamics.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        if bounds is None:
            bounds = self.bounds

        x,y = np.meshgrid(
            np.linspace(bounds[0][0],bounds[0][1],N),
            np.linspace(bounds[1][0],bounds[1][1],M))
        X = np.stack((x,y), axis=-1)

        grad = self.grad(X)
        u = grad[...,0]
        v = grad[...,1]
        ax.streamplot(x,y,u,v)

        ax.set_title(str(self))
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # self.plot_all_critical_points(ax)

        if show_legend:
            ax.legend(loc='lower left')

        return fig, ax


class CovarianceLearning(object):
    def __init__(self, Sigma, dim=2):
        self.dim = dim
        self.Sigma = Sigma
    
    def sample(self, size):
        x = np.random.multivariate_normal(
            mean = np.zeros(self.dim),
            cov = self.Sigma, size=size)
        z = np.random.multivariate_normal(
            mean = np.zeros(self.dim), 
            cov = np.eye(self.dim), size=size)
        return x,z

    def grad(self, V, W, minibatch=50, lambda_reg=0.0):
        x,z = self.sample(minibatch)

        nabla_v = np.zeros((self.dim, self.dim))
        nabla_w = np.zeros((self.dim, self.dim))
        for i in range(minibatch):
            x_vec = x[i].reshape(-1,1)
            Sigma_x = np.dot(x_vec, x_vec.T)
            z_vec = z[i].reshape(-1,1)
            Sigma_z = np.dot(z_vec, z_vec.T)
            nabla_w += Sigma_x - np.dot(np.dot(V, Sigma_z), V.T) - 2*lambda_reg*W
            nabla_v += np.dot(np.dot(W + W.T, V), Sigma_z) - 2*lambda_reg*V
        nabla_w = -nabla_w/minibatch
        nabla_v = -nabla_v/minibatch

        return [nabla_v, nabla_w]


class KellyAuction(Problem):
    def __init__(self, G, Q=1, Z=1):
        self.G = G
        self.Q = Q
        self.Z = Z
        self.pick_adv_matrix = np.ones((len(G), len(G))) - np.identity(len(G))
    
    def dim(self):
        return len(self.G)

    def f(self, X):
        return (self.Q*self.G*X)/(self.Z + np.sum(X, aix=-1)) - X

    def grad(self, X):
        shape = X.shape
        X = X.reshape(-1, shape[-1])
        X = X.swapaxes(0, -1)
        grad = self.pick_adv_matrix.dot(X) + self.Z
        grad = grad / (np.sum(X, axis=0) + self.Z)**2
        grad = grad.swapaxes(0, -1)
        grad = (self.G * self.Q * grad)
        grad = 1 - grad
        return grad.reshape(shape)


class NoiseProfile(Problem):
    def __init__(self, sigma_abs, sigma_rel) -> None:
        self.sigma_abs = sigma_abs
        self.sigma_rel = sigma_rel

    def add_noise(self, grad):
        return grad + self.sigma_abs * np.random.normal(size=grad.shape) + self.sigma_rel * np.linalg.norm(grad, axis=-1, keepdims=True) * np.random.normal(size=grad.shape)


class KellyAuction2D(KellyAuction):
    def __init__(self):
        super().__init__(G=np.array([2.0, 3.0]), Q=1.0, Z=1.0)
        self.equilibrium = np.array([0.1396, 0.7094])


class KellyAuction4D(KellyAuction):
    def __init__(self):
        super().__init__(G=np.array([1.8, 2.0, 2.2, 2.4]), Q=1000, Z=100)
        self.equilibrium = np.array([185.76, 326.15, 441.015, 536.735])
