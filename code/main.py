import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time


def kronecker_product(A, B):
    """Compute the Kronecker product of two matrices."""
    return np.kron(A, B)


def project_to_first_quadrant(z):
    """Project a vector to the first quadrant (non-negative orthant)."""
    return np.maximum(z, 0)


def project_to_omega(y):
    """Project to the set Ω_i: 2-norm ≤ 5 for each agent's variables."""
    n = len(y) // 2  # Number of agents
    y_reshaped = y.reshape(n, 2)
    norm = np.linalg.norm(y_reshaped, axis=1)
    scale = np.minimum(1, 5 / np.maximum(norm, 1e-10))
    return (y_reshaped * scale[:, np.newaxis]).reshape(-1)


def subgradient_f(x):
    """Compute a subgradient of the objective functions f_i for each agent,
    using uniform random selection for non-differentiable points."""
    n = len(x) // 2  # Number of agents (4)
    grad = np.zeros_like(x)

    for i in range(n):
        idx1, idx2 = 2 * i, 2 * i + 1
        x_i1, x_i2 = x[idx1], x[idx2]

        if i == 0:  # f_1(x_1) = 3x_11 - 2x_12
            grad[idx1] = 3
            grad[idx2] = -2

        elif i == 1:  # f_2(x_2) = x_21^2 + x_22^2
            grad[idx1] = 2 * x_i1
            grad[idx2] = 2 * x_i2

        elif i == 2:  # f_3(x_3) = |x_31| + |x_32|
            grad[idx1] = (
                1 if x_i1 > 0 else (-1 if x_i1 < 0 else np.random.uniform(-1, 1))
            )
            grad[idx2] = (
                1 if x_i2 > 0 else (-1 if x_i2 < 0 else np.random.uniform(-1, 1))
            )

        elif i == 3:  # f_4(x_4) = max(0, x_41) + max(0, x_42)
            grad[idx1] = (
                1 if x_i1 > 0 else (0 if x_i1 < 0 else np.random.uniform(0, 1))
            )
            grad[idx2] = (
                1 if x_i2 > 0 else (0 if x_i2 < 0 else np.random.uniform(0, 1))
            )

    return grad


# Global variables to track real time
computation_start_time = None
real_time_stamps = []


def system_dynamics(t, state, L, C, D, b):
    """Define the system dynamics based on the given equations."""
    global computation_start_time, real_time_stamps

    # Record real computation time at each evaluation
    if computation_start_time is not None:
        real_time_stamps.append(time.time() - computation_start_time)

    n_x = len(D[0])  # Dimension of x (and y, gamma): 8
    n_z = len(C[0])  # Dimension of z (and beta): 8

    # Extract state variables
    y = state[:n_x]
    z = state[n_x:n_x + n_z]
    alpha = state[n_x + n_z:2 * n_x + n_z]
    beta = state[2 * n_x + n_z:2 * n_x + 2 * n_z]
    gamma = state[2 * n_x + 2 * n_z:3 * n_x + 2 * n_z]

    # Compute x = P_Ω(y)
    x = project_to_omega(y)

    # Compute L_m = L ⊗ I_2
    I2 = np.eye(2)
    L_m = kronecker_product(L, I2)

    # Compute projections
    z_tilde = project_to_first_quadrant(z)
    beta_tilde = project_to_first_quadrant(beta)

    # Compute derivatives
    dy_dt = x - subgradient_f(x) - L_m @ alpha + gamma - y
    dz_dt = -z + z_tilde - C @ beta_tilde - D @ gamma
    dalpha_dt = L_m @ x
    dbeta_dt = -beta + beta_tilde + C.T @ z_tilde - b
    dgamma_dt = D.T @ z_tilde - x

    return np.concatenate([dy_dt, dz_dt, dalpha_dt, dbeta_dt, dgamma_dt])


def evaluate_objective(x):
    """Evaluate the total objective function value."""
    n = len(x) // 2
    obj_value = 0

    for i in range(n):
        idx1, idx2 = 2 * i, 2 * i + 1
        x_i1, x_i2 = x[idx1], x[idx2]

        if i == 0:  # f_1
            obj_value += 3 * x_i1 - 2 * x_i2
        elif i == 1:  # f_2
            obj_value += x_i1 ** 2 + x_i2 ** 2
        elif i == 2:  # f_3
            obj_value += abs(x_i1) + abs(x_i2)
        elif i == 3:  # f_4
            obj_value += max(0, x_i1) + max(0, x_i2)

    return obj_value


def setup_and_solve_system():
    """Set up and solve the differential equation system."""
    global computation_start_time, real_time_stamps

    # Define the Laplacian matrix L
    L = np.array([
        [2, -1, -1, 0],
        [-1, 1, 0, 0],
        [-1, 0, 2, -1],
        [0, 0, -1, 1]
    ])

    # System parameters
    n = 4  # Number of agents
    d = 2  # Dimension of each agent's variable

    # Define D matrix: D = diag(D_1, ..., D_4), where D_i = I_2
    D = np.eye(n * d)

    # Define C matrix and b vector for inequality constraints
    C = np.zeros((n * d, n * d))
    b = np.zeros(n * d)

    # Set up constraints: e_i^T z_i ≤ c_i
    C[0:2, 0:2] = np.array([[1, 0], [0, 1]])
    b[0:2] = 10
    C[2:4, 2:4] = np.array([[2, 0], [0, 1]])
    b[2:4] = 12
    C[4:6, 4:6] = np.array([[1, 0], [0, 2]])
    b[4:6] = 12
    C[6:8, 6:8] = np.array([[2, 0], [0, 2]])
    b[6:8] = 16

    # Initial conditions
    y0 = np.random.uniform(-10, 10, n * d)
    z0 = np.zeros(n * d)
    alpha0 = np.zeros(n * d)
    beta0 = np.zeros(n * d)
    gamma0 = np.zeros(n * d)

    initial_state = np.concatenate([y0, z0, alpha0, beta0, gamma0])

    # Time span for integration
    t_span = (0, 42.5)
    t_eval = np.linspace(0, 42.5, 85)

    # Reset and start timing
    real_time_stamps = []
    computation_start_time = time.time()

    # Solve the system
    solution = solve_ivp(
        lambda t, state: system_dynamics(t, state, L, C, D, b),
        t_span,
        initial_state,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-5,
        atol=1e-5,
    )

    total_computation_time = time.time() - computation_start_time

    # Compute x(t) = P_Ω(y(t))
    y_values = solution.y[:n * d, :]
    x_values = np.array([project_to_omega(y_values[:, i]) for i in range(len(solution.t))]).T

    # Print computation statistics
    print(f"Total computation time: {total_computation_time:.4f} seconds")
    print(f"Number of function evaluations: {solution.nfev}")
    print(f"Average time per evaluation: {total_computation_time / solution.nfev * 1000:.4f} ms")

    return solution, n, d, x_values, total_computation_time


def plot_results(solution, n, d, x_values, computation_time):
    """Plot: objective value over time, final x values, and L2 distance to optimum over time."""
    plt.rcParams['font.family'] = 'Times New Roman'

    # Use real computation time instead of integration time
    # Distribute the total time evenly across evaluation points
    t_real = np.linspace(0, computation_time, len(solution.t))
    t_max = t_real[-1]  # 最后时刻
    t_max_rounded = round(t_max, 2)  # 四舍五入到小数点后2位

    # Compute objective values over time
    obj_values = np.array([evaluate_objective(x_values[:, i]) for i in range(len(solution.t))])
    abs_obj_values = np.abs(obj_values)

    # Plot 1: Objective function convergence
    plt.figure(figsize=(10, 6))
    plt.plot(t_real, abs_obj_values, linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(r'$|f(x)-f(x^*)|$', fontsize=12)
    plt.xlim(left=0, right=t_max * 1.02)  # 右边留一点空间
    plt.ylim(bottom=0)

    # 添加最后时刻到 x 轴刻度（四舍五入）
    current_ticks = plt.gca().get_xticks()
    # 如果最后时刻不在当前刻度中，添加它
    if t_max_rounded not in current_ticks:
        new_ticks = np.append(current_ticks[current_ticks < t_max_rounded], t_max_rounded)
        plt.xticks(new_ticks)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig("p1.png", dpi=600)

    # Plot 2: L2 distance to [0, 0] over time
    plt.figure(figsize=(10, 8))
    for i in range(n):
        idx1 = 2 * i
        idx2 = 2 * i + 1
        l2_norm = np.sqrt(x_values[idx1, :] ** 2 + x_values[idx2, :] ** 2)
        plt.plot(t_real, l2_norm, label=fr'$||x_{{{i + 1}}}(t)-x_{{{i + 1}}}^*||_2$')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(r'$||x_i(t)-x_i^*||_2$', fontsize=12)
    plt.xlim(left=0, right=t_max * 1.02)
    plt.ylim(bottom=0)

    current_ticks = plt.gca().get_xticks()
    if t_max_rounded not in current_ticks:
        new_ticks = np.append(current_ticks[current_ticks < t_max_rounded], t_max_rounded)
        plt.xticks(new_ticks)

    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("p3.png", dpi=600)

    plt.show()

    return final_x, obj_values[-1]

# Main execution
if __name__ == "__main__":
    solution, n, d, x_values, comp_time = setup_and_solve_system()
    final_x, final_obj = plot_results(solution, n, d, x_values, comp_time)

    print("\nFinal Solution:")
    for i in range(n):
        print(f"x{i + 1} = {final_x[i]}")
    print(f"Final Objective Value: {final_obj}")

    # Theoretical optimal solution
    theoretical_x = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    theoretical_obj = evaluate_objective(theoretical_x.flatten())
    print("\nTheoretical Optimal Solution:")
    for i in range(n):
        print(f"x{i + 1} = {theoretical_x[i]}")
    print(f"Theoretical Optimal Objective Value: {theoretical_obj}")
