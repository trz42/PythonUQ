import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Part 1: Integral of sin(pi * x)
# ==========================================
print("--- Part 1: Integral of sin(pi * x) ---")

def f(x):
    return np.sin(np.pi * x)

M = 10000  # Number of samples
approximate_integral = 0.0

# Using a loop to match the original script's logic
for i in range(M):
    x = np.random.rand()  # Sample uniformly in [0, 1)
    approximate_integral += f(x)

approximate_integral /= M  # Average over samples
print(f"Estimated integral of sin(pi * x) over [0, 1]: {approximate_integral}")
print(f"Exact integral: {2 / np.pi}")
print()


# ==========================================
# Part 2: Estimate value of Pi
# ==========================================
print("--- Part 2: Estimate value of Pi ---")

def I(x, y):
    return 1.0 if x**2 + y**2 <= 1 else 0.0

M = 10000
approximate_pi = 0.0

for i in range(M):
    # uniform(-1, 1) in numpy
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    approximate_pi += I(x, y)

approximate_pi = (approximate_pi / M) * 4  # Scale by area
print(f"Estimated value of pi: {approximate_pi}")
print(f"Exact value of pi: {np.pi}")
print()


# ==========================================
# Part 3: Convergence Analysis & Plotting
# ==========================================
print("--- Part 3: Convergence Analysis ---")

# 2^(4:15) in Python equivalent
M_values = 2 ** np.arange(4, 16)
reruns = 10
errors = np.zeros((len(M_values), reruns))

for j, M in enumerate(M_values):
    for r in range(reruns):
        # Vectorized operation for better performance in Python
        # (Replaces the inner loop 1:M)
        x = np.random.uniform(-1, 1, M)
        y = np.random.uniform(-1, 1, M)
        
        # Count points inside the unit circle
        # (x^2 + y^2 <= 1) creates a boolean array, sum() counts Trues
        hits = np.sum(x**2 + y**2 <= 1)
        
        current_approx_pi = (hits / M) * 4
        errors[j, r] = abs(current_approx_pi - np.pi)

# Calculate statistics along axis 1 (rows are M values, cols are reruns)
mean_errors = np.mean(errors, axis=1)
std_errors = np.std(errors, axis=1)

# Theoretical Standard Deviation Calculation
# Variance of Bernoulli trial p(1-p) where p = pi/4
variance = (np.pi / 4) - (np.pi**2 / 16)
theoretical_std = 4 * np.sqrt(variance / M_values)

# Plotting
plt.figure(figsize=(10, 6))

# Log-Log plot with error bars
plt.errorbar(M_values, mean_errors, yerr=std_errors, fmt='-o', 
             label='Mean absolute error with std dev', capsize=5)

# Plot theoretical standard deviation
plt.plot(M_values, theoretical_std, linestyle='--', linewidth=2, 
         label='Theoretical std dev', color='orange')

plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("Number of samples M")
plt.ylabel("Absolute error")
plt.title("Convergence of Monte Carlo estimate of pi")
plt.legend(loc='upper right')
plt.grid(True, which="both", ls="-", alpha=0.4)

# Adjust Y limits similar to the Julia script
plt.ylim(bottom=np.min(mean_errors) / 4, top=np.max(mean_errors) * 2)

plt.show()
