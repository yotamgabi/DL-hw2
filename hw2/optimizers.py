import abc
import torch
from torch import Tensor


class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """

    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Layers, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()


class VanillaSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            #  Update the gradient according to regularization and then
            #  update the parameters tensor.
            # ====== YOUR CODE: ======

            # Update the gradient according to regularization
            dp += self.reg * p

            # Update the parameters tensor
            p -= self.learn_rate * dp
            # ========================


class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======

        ## for self explanation need to be deleted:###
        """
        The velocity for each model parameter is updated based on the current gradient and the previous velocity value,
        using the formula v = momentum * v - learning_rate * gradient. 
        The momentum hyperparameter controls the contribution of the previous velocity value to the current update, 
        while the learning_rate hyperparameter controls the step size of the update.

        By taking into account previous gradients, the velocity can help smooth out fluctuations
        in the gradient and accelerate convergence.
        During each optimization step, the velocity values are updated based on the current gradient
        and the previous velocity value, and then used to update the model parameters.
        This allows the optimizer to take into account previous gradients when updating the model parameters,
        which can help accelerate convergence.
        """
        self.velocities = [torch.zeros_like(p) for p, _ in self.params]
        # ========================

    def step(self):
        for i, (p, dp) in enumerate(self.params):
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # update the parameters tensor based on the velocity. Don't forget
            # to include the regularization term.
            # ====== YOUR CODE: ======
            # Compute the regularization term
            reg_term = self.reg * p

            # Update velocity
            self.velocities[i].mul_(self.momentum).add_(self.learn_rate * (dp + reg_term))

            # Update parameters
            p.sub_(self.velocities[i])
            # ========================


class RMSProp(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, decay=0.99, eps=1e-8):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param decay: Gradient exponential decay factor
        :param eps: Constant to add to gradient sum for numerical stability
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.decay = decay
        self.eps = eps

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        # Initialize the per-parameter squared gradient tensor
        self.r = [torch.zeros_like(p) for p, _ in self.params]
        # ========================

    def step(self):
        for i, (p, dp) in enumerate(self.params):
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # Create a per-parameter learning rate based on a decaying moving
            # average of it's previous gradients. Use it to update the
            # parameters tensor.
            # ====== YOUR CODE: ======
            # Update the squared gradient tensor
            self.r[i] = self.decay * self.r[i] + (1 - self.decay) * dp.pow(2)
            # compute the per-parameter learning rate
            lr = self.learn_rate / torch.sqrt(self.r[i] + self.eps)
            # Update the value of the current parameter:
            p[:] = p - lr * dp - self.learn_rate * self.reg * p
            # zero the gradient:
            dp.zero_()
            # ========================
