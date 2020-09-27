import numpy as np
from scipy.integrate import solve_ivp

class DynamicalSystem():

    def __init__(self, fun, x0, t0=0, dt=0.02):
        self.fun = fun
        self.dt = dt

        x0 = np.array(x0)
        t0 = np.array(t0)
        self.sim_data = np.array([t0, *x0])
        self.sim_data = self.sim_data[np.newaxis, :]

    # Propagate the system 't_plus' seconds into the future from the current state
    # Values are returned after each 'self.df' seconds, but the solver's internal step size is adaptive (RK45)
    def propagate(self, t_plus):
        # Define 'x0' as last state and 't0' as last time
        x0 = self.sim_data[-1, 1:]
        t0 = self.sim_data[-1, 0]
        # Define spanned time 't_span' and the times 't_eval' at which state should be evaluated
        t_span = (t0, t0 + t_plus)
        t_eval = np.arange(t0, t0 + t_plus + self.dt, self.dt)
        # Solve the ODE
        sol = solve_ivp(self.fun, t_span, x0, t_eval=t_eval)
        # Save the results
        x = sol.y.T[1:, :]
        t = sol.t[1:]
        t = t[:, np.newaxis]
        new_sim_data = np.concatenate((t, x), axis=1)
        self.sim_data = np.concatenate((self.sim_data, new_sim_data))
        return sol