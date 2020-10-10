import numpy as np

def lorenz_equation_p(params=None):
    if params is None:
        params = {'gamma': 10, 'rho': 28, 'beta': 8. / 3}

    def lorenz_equation(t, x):
        xd = np.zeros(3)
        xd[0] = params['gamma'] * (x[1] - x[0])
        xd[1] = x[0] * (params['rho'] - x[2]) - x[1]
        xd[2] = x[0] * x[1] - params['beta'] * x[2]
        return xd

        #dx1 = g*x2 - g*x1
        #dx2 = r*x1 - r*x3
        #dx3 = x1*x2 - b*x3

    return lorenz_equation
