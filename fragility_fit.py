import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

depths = np.array([0.0, 0.2, 0.8, 1.0, 1.2, 1.8])
x = np.linspace(0.001, 2.0, 500)

# Different sigmas for comparison
sigmas = [0.3,0.2,0.1, 0.05, 0.02, 0.01]
mu = 0  # log-mean corresponding to median at 1

plt.figure(figsize=(8,5))
for sigma in sigmas:
    cdf = lognorm.cdf(x, s=sigma, scale=np.exp(mu))
    plt.plot(x, cdf, label=f'σ={sigma}')

plt.axvline(1, color='k', linestyle='--', label='Depth = 1')
plt.xlabel('Depth')
plt.ylabel('Cumulative probability')
plt.title('Lognormal CDF approaching a step at depth=1')
plt.legend()
plt.grid(True)
plt.show()
