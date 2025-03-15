import numpy as np
from F import LinearFunction,QuadraticFunction,AbsoluteValueFunction,ReLUFunction,NegativeExponentialFunction
from CR import VectorOptimizer



if __name__ == '__main__':

    # Objective Functions
    f_1 = LinearFunction()
    f_2 = QuadraticFunction()
    f_3 = AbsoluteValueFunction()
    f_4 = ReLUFunction()
    f_5 = NegativeExponentialFunction()
    functions = [f_1, f_2, f_3, f_4, f_5]

    # Laplacian Matrix
    L = np.array([
        [2, -1, 0, 0, -1],
        [-1, 2, -1, 0, 0],
        [0, -1, 3, -1, -1],
        [0, 0, -1, 2, -1],
        [-1, 0, -1, -1, 3]
    ])

    # Omega
    Omega_l = 0
    Omega_r = 500
    Omega = [Omega_l, Omega_r]

    # Robust Constraints
    b = np.array([1, 1, 1, 1, 1])
    C = np.array([1, 1, 1, 1, 1])
    D = np.array([1, 1, 1, 1, 1])

    # optimizer
    optimizer = VectorOptimizer(
        objective_functions=functions,
        dimension=1,
        L=L,
        b=b,
        C=C,
        D=D,
        Omega=Omega,
        learning_rate=1e-2,
        max_iterations=1000,
        tolerance=1e-8
    )

    # initial values
    y0 = np.array([10,5,1000,100,10])
    z0 = np.array([1,2,3,4,5])
    alpha0 = np.array([1,2,3,4,5])
    beta0 = np.array([1,2,3,4,5])
    gamma0 = np.array([1,2,3,4,5])

    x_history, f_history = optimizer.optimize(y0,z0,alpha0,beta0,gamma0)

    # result
    print("Initial x:", x_history[0])
    print("Final x:", x_history[-1])
    print("Initial f(y):", f_history[0])
    print("Final f(x):", f_history[-1])
    print("Number of iterations:", len(x_history) - 1)

    # visualize
    optimizer.plot_optimization(x_history, f_history)
