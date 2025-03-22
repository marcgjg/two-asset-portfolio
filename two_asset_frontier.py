import numpy as np
import matplotlib.pyplot as plt

def two_stock_frontier(muA, muB, sigmaA, sigmaB, rho):
    """
    Minimal example that forcibly shows only one 'efficient' point
    if muA == muB, or a segment otherwise.
    No random portfolios, no legend reordering, no Streamlit – just basic Matplotlib.
    """

    # Parametric frontier
    covAB = rho * sigmaA * sigmaB
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

    # MVP
    idx_min = np.argmin(stds)
    mvp_x   = stds[idx_min]
    mvp_y   = rets[idx_min]

    # Decide if returns are 'same'
    tol = 1e-12
    same_return = (abs(muA - muB) < tol)

    # If same returns => entire line is 'inefficient' except MVP
    if same_return:
        mask = np.ones_like(stds, dtype=bool)
        mask[idx_min] = False
        x_inef = stds[mask]
        y_inef = rets[mask]

        x_ef = [mvp_x]
        y_ef = [mvp_y]

        plt.plot(x_inef, y_inef, 'r--', label='Inefficient')
        plt.plot(x_ef, y_ef, 'ro', label='Efficient Frontier (single point)')
        plt.plot(mvp_x, mvp_y, 'k*', label='MVP')  # black star on same point
    else:
        # normal logic
        # check which is higher from the direct user input
        if muA > muB:
            x_inef = stds[:idx_min+1]
            y_inef = rets[:idx_min+1]
            x_ef   = stds[idx_min:]
            y_ef   = rets[idx_min:]
        else:
            x_ef   = stds[:idx_min+1]
            y_ef   = rets[:idx_min+1]
            x_inef = stds[idx_min:]
            y_inef = rets[idx_min:]

        plt.plot(x_inef, y_inef, 'r--', label='Inefficient')
        plt.plot(x_ef, y_ef,   'r-', label='Efficient Frontier')
        plt.plot(mvp_x, mvp_y, 'k*', label='MVP')

    plt.xlabel('Std Dev')
    plt.ylabel('Return')
    plt.title(f"muA={muA}, muB={muB}, sigmaA={sigmaA}, sigmaB={sigmaB}, rho={rho}")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Hard-code two scenarios:

    # ====== CASE 1: same returns => only 1 efficient point ======
    two_stock_frontier(muA=0.03, muB=0.03, sigmaA=0.12, sigmaB=0.15, rho=-0.7)

    # ====== CASE 2: different returns => a segment for the efficient portion ======
    two_stock_frontier(muA=0.03, muB=0.04, sigmaA=0.12, sigmaB=0.15, rho=-0.7)
