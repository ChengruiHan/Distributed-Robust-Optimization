import numpy as np


class Function:
    """
    Base class: Represents a function and its subgradient.
    """
    def value(self, x):
        """
        Compute the function value.
        """
        raise NotImplementedError("Subclasses must implement 'value' method.")

    def subgradient(self, x):
        """
        Compute the subgradient.
        """
        raise NotImplementedError("Subclasses must implement 'subgradient' method.")


class LinearFunction(Function):
    """
    Linear function: f(x) = x
    """
    def value(self, x):
        return x

    def subgradient(self, x):
        return 1  # The gradient is always 1


class QuadraticFunction(Function):
    """
    Quadratic function: f(x) = x^2
    """
    def value(self, x):
        return x**2

    def subgradient(self, x):
        return 2 * x  # The gradient is 2x


class AbsoluteValueFunction(Function):
    """
    Absolute value function: f(x) = |x|
    """
    def value(self, x):
        return np.abs(x)

    def subgradient(self, x):
        if x > 0:
            return 1  # Gradient is 1 when x > 0
        elif x < 0:
            return -1  # Gradient is -1 when x < 0
        else:
            return np.random.uniform(0, 1)  # Subgradient can be any value between [-1, 1] when x = 0


class ReLUFunction(Function):
    """
    ReLU function: f(x) = max(0, x)
    """
    def value(self, x):
        return np.maximum(0, x)

    def subgradient(self, x):
        if x > 0:
            return 1  # Gradient is 1 when x > 0
        elif x < 0:
            return 0  # Gradient is 0 when x < 0
        else:
            return np.random.uniform(0, 1)  # Subgradient can be any value between [0, 1] when x = 0


class NegativeLogFunction(Function):
    """
    Negative logarithm function: f(x) = -ln(x)
    """
    def value(self, x):
        return -np.log(x)

    def subgradient(self, x):
        return -1 / x  # Gradient is -1/x


class NegativeExponentialFunction(Function):
    """
    Negative exponential function: f(x) = exp(-x)
    """
    def value(self, x):
        return np.exp(-x)

    def subgradient(self, x):
        return -np.exp(-x)
