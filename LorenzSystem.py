from DynamicalSystem import DynamicalSystem
from equations.LorenzEquation import lorenz_equation_p


class LorenzSystem(DynamicalSystem):
    def __init__(self, x0, t0=0, dt=0.02,
                 params=None):
        if params is None:
            params = {'gamma': 10, 'rho': 28, 'beta': 8. / 3}

        fun = lorenz_equation_p(params)
        DynamicalSystem.__init__(self, fun, x0, t0, dt)
