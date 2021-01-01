import numpy as np

# https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

def lotka_volterra_equation(params=None):
    if params is None:
        params = {'alpha': 1, 'beta': 3, 'delta': 4, 'gamma': 2}

    def lotka_volterra(t, x, u):
        dx = np.empty(2)
        dx[0] = params['alpha']*x[0] - params['beta']*x[0]*x[1] + u[0]
        dx[1] = params['delta']*x[0]*x[1] - params['gamma']*x[1] + u[1]
        return dx

    return lotka_volterra
