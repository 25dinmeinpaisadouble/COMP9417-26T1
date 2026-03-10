import numpy as np
import matplotlib.pyplot as plt

def sample_cauchy(n, sigma):
    return np.random.standard_cauchy(n) * sigma

def sample_response_contamination(n, d, beta_star, eps, sigma_cauchy, sigma_clean):
    X = np.random.normal(0.0, 1.0, size=(n, d))
    z = np.random.uniform(0.0, 1.0, size=n) < eps
    noise = np.random.normal(0.0, sigma_clean, size=n)
    if np.any(z) and sigma_cauchy > 0:
        noise[z] = sample_cauchy(z.sum(), sigma_cauchy)
    y = X @ beta_star + noise
    return X, y

def sample_covariate_contamination(n, d, beta_star, eps, tau, sigma_clean):
    X = np.random.normal(0.0, 1.0, size=(n, d))
    z = np.random.uniform(0.0, 1.0, size=n) < eps
    if np.any(z):
        X[z] = np.random.normal(0.0, tau, size=(z.sum(), d))
    noise = np.random.normal(0.0, sigma_clean, size=n)
    y = X @ beta_star + noise
    return X, y

## Setup for part (a) 
np.random.seed(42)
n = 1000
d = 20
beta_star = np.ones(d)
sigma_clean = 1.0

epsilons = [0.05, 0.1, 0.2]
sigmas = [1.5, 2, 3]
taus = [1.5, 2, 3]


# Part (a)(i)
print("=" * 60)
print("Part (a)(i)")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle("Part (a)(i)", fontsize=14)

for i, eps in enumerate(epsilons):
    for j, sigma_c in enumerate(sigmas):
        np.random.seed(42)
        X, y = sample_response_contamination(n, d, beta_star, eps, sigma_c, sigma_clean)
        residuals = y - X @ beta_star
        
        ax = axes[i, j]
        ax.scatter(range(n), residuals, alpha=0.4, s=8)
        ax.set_title(f"$\\varepsilon={eps}$, $\\sigma_{{cauchy}}={sigma_c}$", fontsize=11)
        ax.set_xlabel("Index $i$")
        ax.set_ylabel("$y_i - x_i^\\top \\beta^*$")
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()


# Part (a)(ii)
print("=" * 60)
print("Part (a)(ii)")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle("Part (a)(ii)", fontsize=14)

for i, eps in enumerate(epsilons):
    for j, tau in enumerate(taus):
        np.random.seed(42)
        X, y = sample_covariate_contamination(n, d, beta_star, eps, tau, sigma_clean)
        norms = np.linalg.norm(X, axis=1)
        
        ax = axes[i, j]
        ax.hist(norms, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f"$\\varepsilon={eps}$, $\\tau={tau}$", fontsize=11)
        ax.set_xlabel("$\\|x_i\\|_2$")
        ax.set_ylabel("Frequency")

plt.tight_layout()
# plt.show()
# plt.close()



# Part (e)
print("=" * 60)
print("Part (e)")
print("=" * 60)

# Loss functions and gradients
def mse_loss(X, y, beta):
    residuals = y - X @ beta
    return np.mean(residuals**2) / 2

def mse_gradient(X, y, beta):
    n = X.shape[0]
    residuals = y - X @ beta
    return -X.T @ residuals / n

def psi_delta(r, delta):
    return np.where(np.abs(r) <= delta, r, delta * np.sign(r))

def huber_loss(X, y, beta, delta):
    residuals = y - X @ beta
    abs_r = np.abs(residuals)
    loss = np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))
    return np.mean(loss)

def huber_gradient(X, y, beta, delta):
    n = X.shape[0]
    residuals = y - X @ beta
    psi = psi_delta(residuals, delta)
    return -X.T @ psi / n

# GD params
d_e = 20
n_e = 100
beta_star_e = np.ones(d_e)
gamma = 0.05
delta = 1.345
T = 50
num_trials = 10

eps_vals = [0, 0.1, 0.4]
sigma_vals = [0, 1, 5]

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle("Part (e)", fontsize=14)

for row, eps in enumerate(eps_vals):
    for col, sigma_c in enumerate(sigma_vals):
        ax = axes[row, col]
        
        mse_errors_all = np.zeros((num_trials, T + 1))
        huber_errors_all = np.zeros((num_trials, T + 1))
        
        for trial in range(num_trials):
            np.random.seed(42 + trial)
            
            if eps == 0:
                X, y = sample_response_contamination(n_e, d_e, beta_star_e, 0, sigma_c, sigma_clean)
            else:
                X, y = sample_response_contamination(n_e, d_e, beta_star_e, eps, sigma_c, sigma_clean)
            
            #MSE gradient descent
            beta_mse = np.zeros(d_e)
            mse_errors_all[trial, 0] = np.linalg.norm(beta_mse - beta_star_e)
            for t in range(T):
                grad = mse_gradient(X, y, beta_mse)
                beta_mse = beta_mse - gamma * grad
                mse_errors_all[trial, t + 1] = np.linalg.norm(beta_mse - beta_star_e)
            
            #Huber (delta) gradient descent
            beta_huber = np.zeros(d_e)
            huber_errors_all[trial, 0] = np.linalg.norm(beta_huber - beta_star_e)
            for t in range(T):
                grad = huber_gradient(X, y, beta_huber, delta)
                beta_huber = beta_huber - gamma * grad
                huber_errors_all[trial, t + 1] = np.linalg.norm(beta_huber - beta_star_e)
        
        # Plot individual runs
        for trial in range(num_trials):
            ax.plot(range(T + 1), mse_errors_all[trial], color='blue', alpha=0.15, linewidth=0.8)
            ax.plot(range(T + 1), huber_errors_all[trial], color='red', alpha=0.15, linewidth=0.8)
        
        # Plot average with thicker line
        ax.plot(range(T + 1), mse_errors_all.mean(axis=0), color='blue', linewidth=2.5, label='MSE')
        ax.plot(range(T + 1), huber_errors_all.mean(axis=0), color='red', linewidth=2.5, label='Huber ($\\delta$-loss)')
        
        ax.set_title(f"$\\varepsilon={eps}$, $\\sigma={sigma_c}$", fontsize=11)
        ax.set_xlabel("Iteration $t$")
        ax.set_ylabel("$\\|\\beta^{(t)} - \\beta^*\\|_2$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()

# Part (f)
print("=" * 60)
print("Part (f)")
print("=" * 60)

def sgd_mse_gradient(X, y, beta, idx):
    """SGD gradient for MSE using single sample."""
    x_i = X[idx]
    r_i = y[idx] - x_i @ beta
    return -r_i * x_i

def sgd_huber_gradient(X, y, beta, idx, delta):
    """SGD gradient for Huber loss using single sample."""
    x_i = X[idx]
    r_i = y[idx] - x_i @ beta
    psi_r = r_i if np.abs(r_i) <= delta else delta * np.sign(r_i)
    return -psi_r * x_i

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle("Part (f)", fontsize=14)

for row, eps in enumerate(eps_vals):
    for col, sigma_c in enumerate(sigma_vals):
        ax = axes[row, col]
        
        mse_errors_all = np.zeros((num_trials, T + 1))
        huber_errors_all = np.zeros((num_trials, T + 1))
        
        for trial in range(num_trials):
            np.random.seed(42 + trial)
            
            if eps == 0:
                X, y = sample_response_contamination(n_e, d_e, beta_star_e, 0, sigma_c, sigma_clean)
            else:
                X, y = sample_response_contamination(n_e, d_e, beta_star_e, eps, sigma_c, sigma_clean)
            
            #MSE SGD
            beta_mse = np.zeros(d_e)
            mse_errors_all[trial, 0] = np.linalg.norm(beta_mse - beta_star_e)
            
            #random indices for SGD (same for both losses)
            rng = np.random.RandomState(42 + trial + 1000)
            indices = rng.randint(0, n_e, size=T)
            
            for t in range(T):
                idx = indices[t]
                grad = sgd_mse_gradient(X, y, beta_mse, idx)
                beta_mse = beta_mse - gamma * grad
                mse_errors_all[trial, t + 1] = np.linalg.norm(beta_mse - beta_star_e)
            
            #Huber SGD
            beta_huber = np.zeros(d_e)
            huber_errors_all[trial, 0] = np.linalg.norm(beta_huber - beta_star_e)
            for t in range(T):
                idx = indices[t]
                grad = sgd_huber_gradient(X, y, beta_huber, idx, delta)
                beta_huber = beta_huber - gamma * grad
                huber_errors_all[trial, t + 1] = np.linalg.norm(beta_huber - beta_star_e)
        
        # individual runs
        for trial in range(num_trials):
            ax.plot(range(T + 1), mse_errors_all[trial], color='blue', alpha=0.15, linewidth=0.8)
            ax.plot(range(T + 1), huber_errors_all[trial], color='red', alpha=0.15, linewidth=0.8)
        
        # average with thicker line
        ax.plot(range(T + 1), mse_errors_all.mean(axis=0), color='blue', linewidth=2.5, label='MSE')
        ax.plot(range(T + 1), huber_errors_all.mean(axis=0), color='red', linewidth=2.5, label='Huber ($\\delta$-loss)')
        
        ax.set_title(f"$\\varepsilon={eps}$, $\\sigma={sigma_c}$", fontsize=11)
        ax.set_xlabel("Iteration $t$")
        ax.set_ylabel("$\\|\\beta^{(t)} - \\beta^*\\|_2$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()