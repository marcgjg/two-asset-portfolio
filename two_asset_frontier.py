import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Stock parameters:
# Both have exactly the same return, 0.09
# sigmas differ, correlation = 0.2
# --------------------------
muA = 0.09
muB = 0.09
sigmaA = 0.20
sigmaB = 0.30
rho = 0.2

# Compute covariance
covAB = rho * sigmaA * sigmaB

# Generate parametric frontier: w in [0,1]
n_points = 200
weights = np.linspace(0, 1, n_points)

rets = []
stds = []
for w in weights:
    r = w*muA + (1-w)*muB
    v = (w**2)*(sigmaA**2) + ((1-w)**2)*(sigmaB**2) + 2*w*(1-w)*covAB
    rets.append(r)
    stds.append(np.sqrt(v))

rets = np.array(rets)
stds = np.array(stds)

# Identify the Minimum-Variance Portfolio (MVP)
idx_min = np.argmin(stds)
mvp_x = stds[idx_min]
mvp_y = rets[idx_min]

# --------------------------
# Plot everything except MVP as "inefficient"
# and the single MVP point as "efficient frontier."
# --------------------------

# We'll remove the MVP from the 'inefficient' line
mask = np.ones_like(stds, dtype=bool)
mask[idx_min] = False

x_inef = stds[mask]
y_inef = rets[mask]

# 1) Dashed black line for all non-MVP points
plt.plot(x_inef, y_inef, 'k--', label='Inefficient')

# 2) Single red dot for the MVP
#    No line, just a single scatter point
plt.scatter([mvp_x], [mvp_y], color='red', s=80,
            label='Efficient Frontier (single point)')

plt.xlabel('Std Dev')
plt.ylabel('Return')
plt.title('Same Return => Single Efficient Point\n(muA=0.09, muB=0.09, sigmaA=0.20, sigmaB=0.30, rho=0.2)')
plt.legend()
plt.show()
