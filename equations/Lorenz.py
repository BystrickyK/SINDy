import numpy as np

def lorenz_equation(params=None):
    if params is None:
        params = {'gamma': 10, 'rho': 28, 'beta': 8. / 3}

    def lorenz(t, x, u):
        dx = np.zeros(3)
        dx[0] = params['gamma'] * (x[1] - x[0]) + u[0]*12.5
        dx[1] = x[0] * (params['rho'] - x[2]) - x[1] + u[1]*x[1]
        dx[2] = x[0] * x[1] - params['beta'] * x[2] + u[2]*x[2]
        return dx

        #dx1 = g*x2 - g*x1
        #dx2 = r*x1 - r*x3
        #dx3 = x1*x2 - b*x3

    return lorenz
