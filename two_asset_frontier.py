import numpy as np
import matplotlib.pyplot as plt

def plot_frontier(muA, muB, sigmaA, sigmaB, rho):
    """
    If muA == muB, we label the entire parametric line as 'inefficient' 
    except for the single MVP, which is 'efficient'.
    Otherwise, we do the usual logic of splitting at the MVP.
    """

    # 1) Parametric frontier (weights from 0 to 1)
    covAB = rho * sigmaA * sigmaB
    n_points = 200
    weights = np.linspace(0, 1, n_points)

    rets = []
    stds = []
    for w in weights:
        r = w*muA + (1-w)*muB
        v = (w**2)*sigmaA**2 + ((1-w)**2)*sigmaB**2 + 2*w*(1-w)*covAB
        rets.append(r)
        stds.append(np.sqrt(v))

    rets = np.array(rets)
    stds = np.array(stds)

    # 2) MVP
    idx_min = np.argmin(stds)
    mvp_x   = stds[idx_min]
    mvp_y   = rets[idx_min]

    # 3) If muA == muB => single-point efficiency
    tol = 1e-12
    same_return = (abs(muA - muB) < tol)

    if same_return:
        # Everything is the same return, so only MVP is efficient
        # The rest is inefficient
        mask = np.ones_like(stds, dtype=bool)
        mask[idx_min] = False

        x_inef = stds[mask]
        y_inef = rets[mask]

        # MVP alone => 'efficient frontier'
        x_ef = [mvp_x]
        y_ef = [mvp_y]

        plt.plot(x_inef, y_inef, 'k--', label='Inefficient (same return)')
        # Single red point for the MVP
        plt.plot(x_ef, y_ef, 'ro', label='Efficient Frontier (single point)')

    else:
        # normal logic
        stdA, retA = stds[-1], rets[-1]  # w=1 => A
        stdB, retB = stds[0],  rets[0]   # w=0 => B

        if retA > retB:
            # A has higher => from idx_min..end is efficient
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
        plt.plot(x_ef,   y_ef,   'r-',  label='Efficient Frontier')

    # 4) Mark the MVP
    plt.plot(mvp_x, mvp_y, 'k*', ms=10, label='MVP')

    plt.xlabel('Std Dev')
    plt.ylabel('Return')
    title = f'muA={muA}, muB={muB}, sigmaA={sigmaA}, sigmaB={sigmaB}, rho={rho}'
    plt.title(title)
    plt.legend()


if __name__ == '__main__':
    # ============== CASE 1: SAME RETURNS ==============
    plt.figure(figsize=(5,4))

    # EXACT same returns => should see a single red dot for efficient
    # and a dashed black line for everything else
    plot_frontier(muA=0.09, muB=0.09, sigmaA=0.20, sigmaB=0.30, rho=0.2)

    plt.tight_layout()
    plt.show()

    # ============== CASE 2: DIFFERENT RETURNS ==============
    plt.figure(figsize=(5,4))

    # Different returns => we'll see a dashed portion + a solid portion
    plot_frontier(muA=0.09, muB=0.10, sigmaA=0.20, sigmaB=0.30, rho=0.2)

    plt.tight_layout()
    plt.show()
