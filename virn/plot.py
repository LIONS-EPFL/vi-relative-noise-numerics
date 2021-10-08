from itertools import cycle
from virn.logging import Metrics
from virn.problems import KellyAuction
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace


def savefigs(name, figs):
    print(f"Saving '{name}'")
    for prefix, fig in figs.__dict__.items():
        fig.savefig(f"../figs/{name}_{prefix}.pdf")
        fig.savefig(f"../figs/{name}_{prefix}.pdf-1.png", dpi=300)


def reduce_points(points, num_points, xscale='linear'):
    """
    Picks `num_points` from points either according to a logarithmic or linear scale. 
    The distinction is necessary since a linear reduction would be incorrectly displayed on a log plot 
    (there would be very few points to represent the left hand side).
    """
    if xscale == 'log':
        max_exponent = np.log10(points.shape[0]-1)
        p = np.logspace(0, max_exponent, num=num_points, dtype=np.int)
    elif xscale == 'linear':
        p = np.linspace(0, points.shape[0]-1, num=num_points, dtype=np.int)
    else:
        raise ValueError("xscale must be 'log' or 'linear'")
    log_points = points[p]
    return p, log_points


def plot_mean_std(ax, stat, axis=1, num_points=None, ylim_factor=None, **style_kwargs):
    if num_points is None:
        num_points = stat.shape[0]
    
    mean = np.mean(stat, axis=axis)
    x_axis, mean = reduce_points(mean, num_points=num_points, xscale=ax.get_xscale())

    # To avoid rescaling the plot just to capture all the variance for tiny regime (e.g. 10^{-7})
    if ylim_factor is not None:
        assert len(ylim_factor) == 2
        ax.set_ylim(ylim_factor[0]*np.min(mean), ylim_factor[1]*np.max(mean))
    
    ax.plot(x_axis, mean, **style_kwargs)

    if stat.shape[axis] > 1:
        factor = 1
        var = np.var(stat, axis=axis)
        x_axis, var = reduce_points(var, num_points=num_points, xscale=ax.get_xscale())
        top = mean + factor * np.sqrt(var)
        bottom = mean - factor * np.sqrt(var)
        ax.fill_between(x_axis, top, bottom, alpha=.2, color=style_kwargs.get('color', None))


def plot_kelly(kelly: KellyAuction, metrics_dict: Metrics, T, figsize=[3.5, 4], num_points=None, ylim_factor=None):
    """Generate plots for a kelly auction.

    Args:
        kelly (KellyAuction):
        <metrics_dict (Metrics):
        T (int): Total length of metric.
        figsize (list, optional): Defaults to [3.5, 4].
        num_points (int, optional): Total number of point to include in the plot. If None the whole metric is used.
        ylim_factor (List[float], optional): If specified the y axis range is set proportional to the minimum of maximum mean value. It can be used to avoid the standard deviation pointlessly expanding the range. Defaults to None.

    Returns:
        [type]: [description]
    """

    with plt.style.context(("tableau-colorblind10",)):
        mpl.rcParams["figure.figsize"] = figsize

        figs = Namespace()
        axes = Namespace()

        color_cycle = cycler(color=['C1', 'C2', 'C3', 'C4', 'C5'])
        #marker_cycle = cycler(marker=(',', '+', '.', 'o', '*'))
        linestyle_cycle = cycler(linestyle=("-","--","-.",":", "-"))
        style_cycle = iter(color_cycle + linestyle_cycle)
        markevery=T//10
        
        figs.players, axes.players = plt.subplots()
        #axes.players.set_title("Kelly auction players")
        axes.players.set_ylabel("$u^p(\\bar{x}^p_{t+1/2};\\bar{x}^{-p}_{t+1/2})$")
        axes.players.set_xlabel("t")
        axes.players.scatter(T//2 * np.ones_like(kelly.equilibrium), kelly.equilibrium, marker="*", color='black', zorder=5)
        axes.players.set_ylim(0, kelly.equilibrium.max() * 1.5)

        if kelly.dim() == 2:
            figs.traj, axes.traj = plt.subplots()
            kelly.plot_vectorfield(axes.traj, bounds=np.array([[0,1],[0,1]]))
            #axes.traj.set_title("Kelly auction")

        figs.stoc_grad, axes.stoc_grad = plt.subplots()
        axes.stoc_grad.set_yscale("log")
        #axes.stoc_grad.set_xscale("log")
        axes.stoc_grad.set_ylabel("$\\| stochastic\ grad_t \\|$")
        axes.stoc_grad.set_xlabel("t")

        figs.grad, axes.grad = plt.subplots()
        axes.grad.set_yscale("log")
        #axes.grad.set_xscale("log")
        axes.grad.set_ylabel("$\\| grad_t \\|$")
        axes.grad.set_xlabel("t")

        figs.last_iter, axes.last_iter = plt.subplots()
        axes.last_iter.set_yscale("log")
        #axes.last_iter.set_xscale("log")
        axes.last_iter.set_ylabel("$\\| x_{t+1/2}-x^* \\|$")
        axes.last_iter.set_xlabel("t")

        figs.average_iter, axes.average_iter = plt.subplots()
        axes.average_iter.set_yscale("log")
        axes.average_iter.set_xscale("log")
        axes.average_iter.set_ylabel("$\\| \\bar{x}_{t+1/2}-x^* \\|$")
        axes.average_iter.set_xlabel("t")

        for label, metrics in metrics_dict.items():
            style = next(style_cycle)

            # Only take x_half
            trajectory = metrics['trajectory'][::2]
            grads = metrics['grads'][::2]
            stoc_grads = metrics['stoc_grads'][::2]
        
            # axes.traj (1 sample)
            if kelly.dim() == 2:
                axes.traj.plot(trajectory[:,0,0], trajectory[:,0,1], label=label, markevery=markevery, **style)

            # axes.stoc_grad
            stoc_grad_norm = np.linalg.norm(stoc_grads, axis=-1)
            plot_mean_std(axes.stoc_grad, 
                stoc_grad_norm, 
                label=label, 
                num_points=num_points, 
                ylim_factor=ylim_factor, 
                **style)

            # axes.grad
            grad_norm = np.linalg.norm(grads, axis=-1)
            plot_mean_std(axes.grad, 
                grad_norm, 
                label=label, 
                num_points=num_points, 
                ylim_factor=ylim_factor, 
                **style)

            # axes.last_iter
            last_distance = np.linalg.norm(trajectory - kelly.equilibrium, axis=-1)
            plot_mean_std(axes.last_iter, 
                last_distance, 
                label=label, 
                num_points=num_points, 
                ylim_factor=ylim_factor, 
                **style)

            # axes.average_iter
            average_trajectory = np.cumsum(trajectory, axis=0)
            average_trajectory = average_trajectory / (np.arange(average_trajectory.shape[0]) + 1)[:, None, None]
            average_distance = np.linalg.norm(average_trajectory - kelly.equilibrium, axis=-1)
            plot_mean_std(axes.average_iter, 
                average_distance, 
                label=label, 
                num_points=num_points, 
                ylim_factor=ylim_factor, 
                **style)

            for i in range(kelly.dim()):
                l = label if i == 0 else None
                plot_mean_std(axes.players, 
                    average_trajectory[...,i], 
                    label=l,
                    num_points=num_points, 
                    **style)

        for ax in axes.__dict__.values():
            ax.legend(loc='upper right')
        #axes.players.legend(loc='upper right')

        for fig in figs.__dict__.values():
            fig.tight_layout()

        return figs, axes


def plot_trajectory(ax, trajectory, loss,
    title=None, 
    full_contour=False, 
    max_trajectories=5, 
    bounds=None, 
    colors=None
):
    if colors is None:
        colors = cycle(['red'])

    if full_contour:
        bounds = np.array([
            [np.min(trajectory[...,0]), np.max(trajectory[...,0])],
            [np.min(trajectory[...,1]), np.max(trajectory[...,1])]])
    elif bounds is None:
        bounds = loss.bounds

    loss.plot_vectorfield(ax=ax, bounds=bounds, show_legend=False)

    # Cap at plotting 5 trajectories
    n = min(trajectory.shape[1], max_trajectories)

    # Plot trajectories
    for i in range(n):
        color = next(colors)
        ax.plot(trajectory[:,i, 0], trajectory[:,i, 1], color=color)
        ax.scatter(trajectory[0,i,0], trajectory[0,i,1], color=color)

    # plt.legend(loc='lower left')
    # plt.colorbar(m)
    
    if title:
        ax.set_title(title)


def plot_2d_gaussian(mu, Sigma, title):
    """Pulled from https://github.com/vsyrgkanis/optimistic_GAN_training
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(-3, 4, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure(figsize=(7,7))
    plt.title(title)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)


def plot_covariance_weights(W, V, Sigma, T, save_prefix=None):
    """Pulled from https://github.com/vsyrgkanis/optimistic_GAN_training
    """
    dim = np.shape(W)[1]
    plt.figure(figsize=(5, 5))
    plt.rc('text', usetex=True)
    plt.suptitle("Discriminator Weights")
    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, i*dim+j+1)
            plt.plot(W[:, i, j], label="$W_{"+str(i)+str(j)+"}$")
            plt.legend()
    if save_prefix is not None:
        plt.savefig(f"../figs/{save_prefix}_covariance_dis.pdf")
    plt.show()


    plt.figure(figsize=(5, 5))
    plt.rc('text', usetex=True)
    plt.suptitle("Generator Weights")
    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, i*dim+j+1)
            plt.plot(V[:, i, j], label="$V_{"+str(i)+str(j)+"}$")
            plt.legend()
    if save_prefix is not None:
        plt.savefig(f"../figs/{save_prefix}_covariance_gen.pdf")
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.rc('text', usetex=True)
    plt.suptitle("Generator Sigma")
    Sigma_G = np.zeros((T,dim,dim))
    for t in range(T):
        Sigma_G[t] = np.dot(V[t], V[t].T)
    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, i*dim+j+1)
            plt.plot(Sigma_G[:, i, j], label="$\Sigma^G_{"+str(i)+str(j)+"}$")
            plt.legend()
    if save_prefix is not None:
        plt.savefig(f"../figs/{save_prefix}_covariance_Sigma.pdf")
    plt.show()

    print("Last Iteration Sigma:{}".format(Sigma_G[-1]))
    print("Sigma of Average Weights: {}".format(np.dot(np.mean(V, axis=0), np.mean(V, axis=0).T)))
    print("Average Sigma: {}".format(np.mean(Sigma_G, axis=0)))
    print("True Sigma: {}".format(Sigma))
    
    x = np.random.multivariate_normal(mean = np.zeros(dim),
                                  cov = Sigma, size=1000)
    x_G = np.random.multivariate_normal(mean = np.zeros(dim),
                                      cov = np.mean(Sigma_G, axis=0), size=1000)
    x_G_last = np.random.multivariate_normal(mean = np.zeros(dim),
                                      cov = Sigma_G[-1], size=1000)
    plt.figure(figsize=(10,5))
    plt.suptitle("True Distribution vs Generators distribution")
    if dim == 2:
        plt.subplot(1,3,1)
        plt.title("True Distribution")
        plt.scatter(x[:,0], x[:,1])
        plt.subplot(1,3,2)
        plt.title("Uniform Random Generator")
        plt.scatter(x_G[:,0], x_G[:,1])
        plt.subplot(1,3,3)
        plt.title("Last Iterate Generator")
        plt.scatter(x_G_last[:,0], x_G_last[:,1])
    if dim == 1:
        plt.subplot(1,3,1)
        plt.title("True Distribution")
        plt.hist(x)
        plt.subplot(1,3,2)
        plt.title("Uniform Random Generator")
        plt.hist(x_G)
        plt.subplot(1,3,3)
        plt.title("Last Iterate Generator")
        plt.hist(x_G_last)
    plt.show()
    
    if dim==2:
        plot_2d_gaussian(np.zeros(dim), Sigma, "True Distribution")
        if save_prefix is not None:
            plt.savefig(f"../figs/{save_prefix}_covariance_true.pdf")
            plt.show()
        
        for i in [-1, -20, -35, -50, -70]:
            plot_2d_gaussian(np.zeros(dim), Sigma_G[i], "Last Iterate Generator Distribution")
            if save_prefix is not None:
                plt.savefig(f"../figs/{save_prefix}_covariance_iterate_{i}.pdf")
                plt.show()
