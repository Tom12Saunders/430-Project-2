import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given parameters
beta = 0.0075  # delayed neutron fraction
lambda_ = 0.08  # mean delayed neutron precursor lifetime
Lambda_ = 6e-5  # mean neutron lifetime
n0 = 1  # neutron density
C0 = beta * n0 / (Lambda_ * lambda_)  # initial delayed neutron precursor density

# Analytic solutions
def analytic_negative_n(t):
    return n0 * (0.869546 * np.exp(-958.344 * t) + 0.130454 * np.exp(-0.0695645 * t))

def analytic_negative_C(t):
    return n0 * (0.0113427 * np.exp(-958.344 * t) + 1562.61 * np.exp(-0.0695645 * t))

def analytic_positive_n(t):
    return n0 * (-0.2495 * np.exp(-100.1 * t) + 1.24950 * np.exp(0.01998 * t))

def analytic_positive_C(t):
    return n0 * (0.311814 * np.exp(-100.1 * t) + 1562.19 * np.exp(0.01998 * t))

# Separate backward Euler functions for Neutron & Neutron Precursor densities for both positive and negative reactivity insertions.
def backward_euler_positive_n(t, dt):
    n_values = [n0]
    C_current = C0
    for i in range(1, len(t)):
        n_next = (n_values[-1] + dt * (rho_positive - beta) * n_values[-1] + beta * dt * C_current) / (1 + dt * (rho_positive - beta))
        n_values.append(n_next)
        C_current = (C_current + dt * lambda_ * n_values[-1]) / (1 + dt * lambda_)
    return n_values

def backward_euler_negative_n(t, dt):
    n_values = [n0]
    C_current = C0
    for i in range(1, len(t)):
        n_next = (n_values[-1] + dt * (rho_negative - beta) * n_values[-1] + beta * dt * C_current) / (1 + dt * (rho_negative - beta))
        n_values.append(n_next)
        C_current = (C_current + dt * lambda_ * n_values[-1]) / (1 + dt * lambda_)
    return n_values

def backward_euler_positive_C(t, dt):
    C_values = [C0]
    n_current = n0
    for i in range(1, len(t)):
        n_next = (n_current + dt * (rho_positive - beta) * n_current + beta * dt * C_values[-1]) / (1 + dt * (rho_positive - beta))
        C_next = (C_values[-1] + dt * lambda_ * n_next) / (1 + dt * lambda_)
        C_values.append(C_next)
        n_current = n_next
    return C_values

def backward_euler_negative_C(t, dt):
    C_values = [C0]
    n_current = n0
    for i in range(1, len(t)):
        n_next = (n_current + dt * (rho_negative - beta) * n_current + beta * dt * C_values[-1]) / (1 + dt * (rho_negative - beta))
        C_next = (C_values[-1] + dt * lambda_ * n_next) / (1 + dt * lambda_)
        C_values.append(C_next)
        n_current = n_next
    return C_values

t_neg = np.arange(0, 30 + 0.15, 0.15)
n_neg = np.zeros_like(t_neg)
C_neg = np.zeros_like(t_neg)
n_neg[0] = n0
C_neg[0] = C0

t_pos = np.arange(0, 30 + 0.5, 0.5)
n_pos = np.zeros_like(t_pos)
C_pos = np.zeros_like(t_pos)
n_pos[0] = n0
C_pos[0] = C0

# Negative reactivity insertion
rho_neg = -0.05  # reactivity for negative insertion

for i in range(len(t_neg) - 1):
    dt = t_neg[i+1] - t_neg[i]
    # Compute using backward Euler method
    n_neg[i+1] = (n_neg[i] + dt * lambda_ * C_neg[i]) / (1 - dt * (rho_neg - beta) / Lambda_)
    C_neg[i+1] = (C_neg[i] + dt * beta * n_neg[i] / Lambda_) / (1 + dt * lambda_)

# Positive reactivity insertion
rho_pos = 0.0015  # reactivity for positive insertion

for i in range(len(t_pos) - 1):
    dt = t_pos[i+1] - t_pos[i]
    # Compute using backward Euler method
    n_pos[i+1] = (n_pos[i] + dt * lambda_ * C_pos[i]) / (1 - dt * (rho_pos - beta) / Lambda_)
    C_pos[i+1] = (C_pos[i] + dt * beta * n_pos[i] / Lambda_) / (1 + dt * lambda_)

# Polynomial fit function
def poly_fit(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the data with a polynomial
params_neg = np.polyfit(t_neg, C_neg, 2)
C_neg_fit = poly_fit(t_neg, *params_neg)

params_pos = np.polyfit(t_pos, C_pos, 2)
C_pos_fit = poly_fit(t_pos, *params_pos)

# Compute the errors between numerical and analytical solutions
error_neg_n = n_neg - analytic_negative_n(t_neg)
error_neg_C = C_neg - analytic_negative_C(t_neg)
error_pos_n = n_pos - analytic_positive_n(t_pos)
error_pos_C = C_pos - analytic_positive_C(t_pos)

# Plotting
plt.figure(figsize=(15, 8))  # Increased figure size

plt.subplot(2, 2, 1)
plt.plot(t_neg, n_neg, label='Numerical')
plt.plot(t_neg, analytic_negative_n(t_neg), '--', label='Analytic: $n_0 (0.869546 \exp(-958.344t) + 0.130454 \exp(-0.0695645t))$')
plt.title(f'Negative Reactivity - n(t)\nError: {np.mean(np.abs(error_neg_n)):.2e}')
plt.legend(fontsize='small')

plt.subplot(2, 2, 2)
plt.plot(t_neg, C_neg, label='Numerical')
plt.plot(t_neg, analytic_negative_C(t_neg), '--', label='Analytic: $n_0 (0.0113427 \exp(-958.344t) + 1562.61 \exp(-0.0695645t))$')
plt.plot(t_neg, C_neg_fit, ':', color='red', label=f'Numerical Polynomial Best Fit: {params_neg[0]:.2f}x^2 + {params_neg[1]:.2f}x + {params_neg[2]:.2f}')
plt.title(f'Negative Reactivity - C(t)\nError: {np.mean(np.abs(error_neg_C)):.2e}')
plt.legend(fontsize='small')

plt.subplot(2, 2, 3)
plt.plot(t_pos, n_pos, label='Numerical')
plt.plot(t_pos, analytic_positive_n(t_pos), '--', label='Analytic: $n_0 (-0.2495 \exp(-100.1t) + 1.24950 \exp(0.01998t))$')
plt.title(f'Positive Reactivity - n(t)\nError: {np.mean(np.abs(error_pos_n)):.2e}')
plt.legend(fontsize='small')

plt.subplot(2, 2, 4)
plt.plot(t_pos, C_pos, label='Numerical')
plt.plot(t_pos, analytic_positive_C(t_pos), '--', label='Analytic: $n_0 (0.311814 \exp(-100.1t) + 1562.19 \exp(0.01998t))$')
plt.plot(t_pos, C_pos_fit, ':', color='red', label=f'Numerical Exponential Best Fit: {params_pos[0]:.2f} exp({params_pos[1]:.2f} t) + {params_pos[2]:.2f}')
plt.title(f'Positive Reactivity - C(t)\nError: {np.mean(np.abs(error_pos_C)):.2e}')
plt.legend(fontsize='small')

plt.tight_layout()
plt.show()

# Forward Euler functions for Neutron & Neutron Precursor densities for negative reactivity insertion.
def forward_euler_negative_n(t, dt):
    n_values = [analytic_negative_n(t[0])]
    C_values = [analytic_negative_C(t[0])]
    for i in range(1, len(t)):
        dn = dt * ((rho_neg - beta) * n_values[-1] + beta * C_values[-1])
        dC = dt * lambda_ * n_values[-1]

        n_values.append(n_values[-1] + dn)
        C_values.append(C_values[-1] + dC)
    return n_values, C_values

# Forward Euler functions for Neutron & Neutron Precursor densities for negative reactivity insertion.
def forward_euler_negative(t, dt):
    n_values = [n0]
    C_values = [C0]
    for i in range(1, len(t)):
        dn = dt * ((rho_neg - beta) * n_values[-1] + beta * C_values[-1] / Lambda_)
        dC = dt * (lambda_ * n_values[-1] - lambda_ * C_values[-1])

        n_values.append(n_values[-1] + dn)
        C_values.append(C_values[-1] + dC)
    return n_values, C_values

# Time range
t_neg_forward = np.arange(25, 30 + 0.15, 0.15)

# Compute using forward Euler method
n_neg_forward, C_neg_forward = forward_euler_negative(t_neg_forward, 0.15)

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t_neg_forward, n_neg_forward, label='Forward Euler')
plt.plot(t_neg_forward, analytic_negative_n(t_neg_forward), '--', label='Analytic')
plt.title('Negative Reactivity - n(t) using Forward Euler')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_neg_forward, C_neg_forward, label='Forward Euler')
plt.plot(t_neg_forward, analytic_negative_C(t_neg_forward), '--', label='Analytic')
plt.title('Negative Reactivity - C(t) using Forward Euler')
plt.legend()

plt.tight_layout()
plt.show()

# Tabulate the numerical solutions
print("t(s)\t\tn(t)\t\t\tC(t)")
print("-"*40)
for t, n, C in zip(t_neg_forward, n_neg_forward, C_neg_forward):
    print(f"{t:.2f}\t\t{n:.6f}\t\t{C:.6f}")
