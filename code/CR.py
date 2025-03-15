import numpy as np
import matplotlib.pyplot as plt
from utils import projection,projection2Omega


class VectorOptimizer:
    def __init__(
            self,
            objective_functions,
            dimension,
            L,
            b,
            C,
            D,
            Omega,
            learning_rate: float = 1,
            max_iterations: int = 10000,
            tolerance: float = 1e-6
    ):
        """
        Initialize the vector optimizer

        Args:
            objective_functions: List of objective functions [f₁(y₁), f₂(y₂), ..., fₙ(yₙ)]
            dimension: The dimension of x_i
            learning_rate: The learning rate
            max_iterations: The maximum number of iterations
            tolerance: Convergence threshold
        """
        self.functions = objective_functions
        self.dim = dimension
        self.Lm = np.kron(L,np.eye(dimension))
        self.b = b
        self.C = C
        self.D = D
        self.Omega = Omega
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.tol = tolerance

    def compute_values(self, y):

        return np.array([f_i.value(y[i]) for i, f_i in enumerate(self.functions)])

    def compute_derivatives(self, y):

        return np.array([f_i.subgradient(y[i]) for i, f_i in enumerate(self.functions)])

    def optimize(self,y0,z0,alpha0,beta0,gamma0):

        y = y0.copy()
        z = z0.copy()
        alpha = alpha0.copy()
        beta = beta0.copy()
        gamma = gamma0.copy()

        x_history = [projection2Omega(y0,self.Omega)]
        f_history = [self.compute_values(x_history[0].copy())]

        for i in range(self.max_iter):

            # Algorithm
            x = projection2Omega(y,self.Omega)
            # print("x",x)

            dy = x - self.compute_derivatives(x) - self.Lm @ alpha + gamma - y
            y_new = y + self.lr * dy
            # print("y",y_new)

            dz = -z + projection(z) - self.C @ projection(beta) - self.D @ gamma
            z_new = z + self.lr * dz
            # print("z",z_new)

            d_alpha = self.Lm @ x
            alpha_new = alpha + self.lr * d_alpha
            # print("alpha",alpha_new)

            d_beta = -beta + projection(beta) + projection(z).T @ self.C - self.b
            beta_new = beta + self.lr * d_beta
            # print("beta",beta_new)

            d_gamma = self.D.T @ projection(z) - x
            gamma_new = gamma + self.lr * d_gamma
            # print("gamma",gamma_new)

            x_history.append(x.copy())
            f_history.append(self.compute_values(x))

            # termination conditions
            if np.max([np.max(np.abs(y_new - y)), np.max(np.abs(z_new - z)),
                          np.max(np.abs(alpha_new - alpha)), np.max(np.abs(beta_new - beta)),
                          np.max(np.abs(gamma_new - gamma))]) < self.tol:
                break

            y = y_new
            z = z_new
            alpha = alpha_new
            beta = beta_new
            gamma = gamma_new

        return x_history, f_history

    def plot_optimization(self, x_history, f_history):

        x_history_array = np.array(x_history)

        # Plot the changes of each component x_i
        plt.figure(figsize=(10, 5))
        for i in range(len(self.functions)):
            plt.plot(x_history_array[:, i], label=f'x_{i + 1}')
        plt.title('x Values over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('x')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot the changes of each component f_i(x_i)
        f_history_array = np.array(f_history)
        plt.figure(figsize=(10, 5))
        for i in range(len(self.functions)):
            plt.plot(f_history_array[:, i], label=f'f_{i + 1}(x_{i + 1})')
        plt.title('Function Values over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
