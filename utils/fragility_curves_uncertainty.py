import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from fragility_curves import fragility_PV

x = np.linspace(0.001, 10.0, 500)

# Different sigmas for comparison
sigmas = [0.3,0.2,0.1, 0.05, 0.02, 0.01]
mu = 1.609

plt.figure(figsize=(8,5))
for sigma in sigmas:
    cdf = lognorm.cdf(x, s=sigma, scale=np.exp(mu))
    plt.plot(x, cdf, label=f'σ={sigma}')

plt.axvline(np.exp(mu), color='k', linestyle='--', label='Depth = 1')
plt.xlabel('Depth')
plt.ylabel('Cumulative probability')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x, fragility_PV(x, 1))
plt.grid(True)
plt.show()
