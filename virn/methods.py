import numpy as np


class Method(object):
    def step(Z, grad):
        raise NotImplementedError


class GDA(Method):
    def __init__(self, lr) -> None:
        self.lr = lr

    def step(self, z, grad):
        return z - self.lr * grad


class OG(Method):
    def __init__(self, lr) -> None:
        self.lr = lr
        self.prev_grad = None

    def step(self, z, grad):
        if self.prev_grad is None:
            self.prev_grad = grad

        z_new = z - 2 * self.lr * grad + self.lr * self.prev_grad

        self.prev_grad = grad
        return z_new


class GeneralizedExtragrad(Method):
    def __init__(self, lr=1.0, adaptive=True) -> None:
        self.lr = lr
        
        # State
        self.has_extrapolated = False
        self.curr_grad_half = None
        self.curr_x_half = None
        self.curr_x = None
        self.curr_y = None
        self.operator_diff_sum = np.zeros(1)
        self.adaptive = adaptive

    def update_step_size(self, grad, grad_half):
        elem = np.sum((grad - grad_half) ** 2, axis=-1)
        self.operator_diff_sum = self.operator_diff_sum + elem

    def step_size(self):
        if self.adaptive:
            return self.lr / np.sqrt(1 + self.operator_diff_sum)[:, None]
        else:
            return self.lr

    def half_step(self, grad):
        self.curr_grad_half = grad
        self.curr_x_half = self.curr_x - self.step_size() * grad
        return self.curr_x_half

    def full_step(self, grad): 
        self.update_step_size(grad, self.curr_grad_half)
        self.curr_y = self.curr_y - grad
        self.curr_x = self.step_size() * self.curr_y
        return self.curr_x

    def step(self, Z, grad):
        if self.curr_y is None:
            self.curr_x_half = Z
            self.curr_y = Z
            self.curr_x = Z

        if not self.has_extrapolated:
            self.has_extrapolated = True
            return self.half_step(grad)
        else:
            self.has_extrapolated = False
            return self.full_step(grad)


class DualX(GeneralizedExtragrad):
    pass


class DualAvg(GeneralizedExtragrad):
    def half_step(self, grad):
        grad = np.zeros_like(grad)
        return super().half_step(grad)


class DualOpt(GeneralizedExtragrad):
    def __init__(self, lr=1.0, adaptive=True) -> None:
        super().__init__(lr=lr, adaptive=adaptive)
        self.prev_grad = None

    def half_step(self, grad):
        return super().half_step(self.prev_grad)

    def full_step(self, grad):
        self.prev_grad = grad
        return super().full_step(grad)

    def step(self, Z, grad):
        if self.prev_grad is None:
            self.prev_grad = grad
        return super().step(Z, grad)
