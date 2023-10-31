import numpy as np
import matplotlib.pyplot as plt

# Given parameters
beta = 0.0075
lambda_ = 0.08
Lambda_ = 6e-5
n0 = 1  # initial neutron density
C0 = beta * n0 / (Lambda_ * lambda_)  # initial delayed neutron precursor density

rho = -0.05  # step negative reactivity
T = 30  # total simulation time


def analytical_n(t):
    return n0 * (0.869546 * np.exp(-958.344 * t) + 0.130454 * np.exp(-0.0695645 * t))


def analytical_C(t):
    return n0 * (0.0113427 * np.exp(-958.344 * t) + 1562.61 * np.exp(-0.0695645 * t))


def forward_euler(h):
    N = int(T / h)
    n = np.zeros(N + 1)
    C = np.zeros(N + 1)
    n[0] = n0
    C[0] = C0

    for i in range(N):
        dn = (rho - beta) / Lambda_ * n[i] + lambda_ * C[i]
        dC = beta / Lambda_ * n[i] - lambda_ * C[i]

        n[i + 1] = n[i] + h * dn
        C[i + 1] = C[i] + h * dC

    return n, C


def backward_euler(h):
    N = int(T / h)
    n = np.zeros(N + 1)
    C = np.zeros(N + 1)
    n[0] = n0
    C[0] = C0

    for i in range(N):
        # Coefficients for the system of equations
        a11 = 1 - h * (rho - beta) / Lambda_
        a12 = -h * lambda_
        a21 = -h * beta / Lambda_
        a22 = 1 + h * lambda_

        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([n[i], C[i]])
        n[i + 1], C[i + 1] = np.linalg.solve(A, b)

    return n, C


def crank_nicholson(h):
    N = int(T / h)
    n = np.zeros(N + 1)
    C = np.zeros(N + 1)
    n[0] = n0
    C[0] = C0

    for i in range(N):
        # Coefficients for the system of equations
        a11 = 1 - 0.5 * h * (rho - beta) / Lambda_
        a12 = -0.5 * h * lambda_
        a21 = -0.5 * h * beta / Lambda_
        a22 = 1 + 0.5 * h * lambda_

        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([n[i] + 0.5 * h * ((rho - beta) / Lambda_ * n[i] + lambda_ * C[i]),
                      C[i] + 0.5 * h * (beta / Lambda_ * n[i] - lambda_ * C[i])])
        n[i + 1], C[i + 1] = np.linalg.solve(A, b)

    return n, C


def trapezoidal_bdf2(h):
    N = int(T / h)
    n = np.zeros(N + 1)
    C = np.zeros(N + 1)
    n[0] = n0
    C[0] = C0

    # Initial step with backward Euler
    a11 = 1 - h * (rho - beta) / Lambda_
    a12 = -h * lambda_
    a21 = -h * beta / Lambda_
    a22 = 1 + h * lambda_
    A = np.array([[a11, a12], [a21, a22]])
    b = np.array([n[0], C[0]])
    n[1], C[1] = np.linalg.solve(A, b)

    for i in range(1, N):
        # Coefficients for the system of equations
        a11 = 3 / 2 - 2 * h * (rho - beta) / Lambda_
        a12 = -2 * h * lambda_
        a21 = -2 * h * beta / Lambda_
        a22 = 3 / 2 + 2 * h * lambda_

        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([2 * n[i] - 0.5 * n[i - 1], 2 * C[i] - 0.5 * C[i - 1]])
        n[i + 1], C[i + 1] = np.linalg.solve(A, b)

    return n, C


def plot_results(t, n, C, method_name):
    plt.figure(figsize=(10, 6))

    # Neutron Density Plot
    plt.subplot(2, 1, 1)
    plt.plot(t, n, label='Numerical', linestyle='-', color='blue')
    plt.plot(t, analytical_n(t), label='Analytical', linestyle='--', color='red')
    plt.title(f'Neutron Density (n) vs Time ({method_name})')
    plt.ylabel('n(t)')
    plt.legend()
    plt.grid(True)

    # Precursor Density Plot
    plt.subplot(2, 1, 2)
    plt.plot(t, C, label='Numerical', linestyle='-', color='blue')
    plt.plot(t, analytical_C(t), label='Analytical', linestyle='--', color='red')
    plt.title(f'Delayed Neutron Precursor Density (C) vs Time ({method_name})')
    plt.xlabel('Time (s)')
    plt.ylabel('C(t)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ********** NOTE: Forward Euler was commented out because it was handled in a separate script with Problem 1.
methods = {
    #"Forward-Euler": forward_euler,
    "Backward-Euler": backward_euler,
    "Crank-Nicholson": crank_nicholson,
    "Trapezoidal/BDF-2": trapezoidal_bdf2
}

time_steps = {
    "Forward-Euler": 0.15,
    "Backward-Euler": 0.15,
    "Crank-Nicholson": 0.001,
    "Trapezoidal/BDF-2": 0.0375
}

for method_name, method_func in methods.items():
    h = time_steps[method_name]
    n, C = method_func(h)
    t = np.linspace(0, T, len(n))
    plot_results(t, n, C, method_name)

# Function to compute the order of accuracy using the Trapezoidal/BDF-2 method
def compute_order_of_accuracy():
    # Time steps for Trapezoidal/BDF-2 method
    time_steps = [0.15, 0.075, 0.0375]
    final_values_n = []
    final_values_C = []

    for h in time_steps:
        n, C = trapezoidal_bdf2(h)
        final_values_n.append(n[-1])
        final_values_C.append(C[-1])

    # Computing the ratios
    Rn = (final_values_n[1] - final_values_n[0]) / (final_values_n[2] - final_values_n[1])
    RC = (final_values_C[1] - final_values_C[0]) / (final_values_C[2] - final_values_C[1])

    # Assuming R = 2^N, we can deduce N as:
    N_n = np.log2(Rn)
    N_C = np.log2(RC)

    print(f"Rn: {Rn:.6f}")
    print(f"RC: {RC:.6f}")
    print(f"Order of accuracy for n between h = 0.15 and h = 0.075: {N_n:.6f}")
    print(f"Order of accuracy for C between h = 0.15 and h = 0.075: {N_C:.6f}")

# Main execution
for method_name, method_func in methods.items():
    h = time_steps[method_name]
    n, C = method_func(h)
    t = np.linspace(0, T, len(n))
    #plot_results(t, n, C, method_name)

# Calculate and display order of accuracy for Trapezoidal/BDF-2 method
compute_order_of_accuracy()