import numpy as np
from scipy.integrate import solve_ivp
from equations.Lorenz import lorenz_equation
from equations.LotkaVolterra import lotka_volterra_equation

class DynamicalSystem():

    def __init__(self, fun, x0, dt=0.01, t0=0, solver='RK45'):
        self.fun = fun
        self.dt = dt



        self.x0 = np.array(x0)
        self.dims = self.x0.shape[0]
        self.t0 = np.array(t0)
        u0 = np.zeros(self.dims)
        self.sim_data = np.hstack([t0, x0, u0])
        self.sim_data = self.sim_data[np.newaxis, :]

        self.solver = solver
    # Propagate the system 't_plus' seconds into the future from the current state
    # Values are returned after each 'self.df' seconds, but the solver's internal step size is adaptive (RK45)
    def propagate(self, t_plus):
        """
        Simulates the dynamic system's natural dynamics (i.e. no external forcing).

        Args:
            t_plus: How many seconds into the future should the system be propagated

        Returns:
            success: Is True if the solver finished without complaining.
        """
        # Define 'x0' as last state and 't0' as last time
        x0 = self.sim_data[-1, 1:self.dims+1]
        t0 = self.sim_data[-1, 0]
        # Define spanned time 't_span' and the times 't_eval' at which state should be evaluated
        t_span = (t0, t0 + t_plus + self.dt/2)
        t_eval = np.arange(t0, t0 + t_plus + self.dt/10000, self.dt)
        # Define unforced system
        fun = lambda t, x: self.fun(t, x, np.zeros([self.dims, 1]).flatten())
        # Solve the ODE
        # TODO: Cross terms between u and x cause the output to be an array instead of scalar for the dimension
        sol = solve_ivp(fun, t_span, x0, t_eval=t_eval, method=self.solver)
        if not sol.success:
            error_str = "Solver failed.\n{sol_msg}".format(sol_msg=sol.message)
            raise RuntimeError(error_str)
        # Save the results
        t = sol.t[1:, np.newaxis]
        x = sol.y[:, 1:]
        u = np.zeros([x.shape[0], t.shape[0]])
        new_sim_data = np.hstack([t, x.T, u.T])
        self.sim_data = np.vstack([self.sim_data, new_sim_data])
        return True

    def propagate_forced(self, t_plus, u_fun):
        """
        Simulates the dynamic system with forcing. Forcing could be interpreted
        as external disturbances, or as a control input.

        Args:
            t_plus: How many seconds into the future should the system be propagated
            u_fun: Tuple of forcing functions. Number of elements of the tuple must be
            equal to number of dimensions of the systems. The index of the function
            (position in the tuple) determines which state is being forced.
            Each function must be in the format fun(t, x)

        Returns:
            success: Is True if the solver finished without complaining.
        """
        # Define 'x0' as last state and 't0' as last time
        x0 = self.sim_data[-1, 1:self.dims+1]
        t0 = self.sim_data[-1, 0]

        # Define spanned time 't_span' and the times 't_eval' at which state should be evaluated
        t_eval = np.arange(t0, t0 + t_plus + self.dt/10000, self.dt)

        # Pre-allocate the array for simulation solutions
        new_sim_data = np.zeros([t_eval.shape[0], self.dims * 2 + 1])
        new_sim_data[:, 0] = t_eval
        new_sim_data[0, 1:self.dims+1] = x0

        # Solve the ODE
        for k, t in enumerate(t_eval[:-1]):
            x_k = new_sim_data[k, 1:self.dims+1]  # state of the system at sample k

            # Evaluate input u
            u_k = np.zeros(self.dims)
            for d in range(self.dims):
                if u_fun[d] is not None:
                    u_k[d] = u_fun[d](t, x_k)
                else:
                    pass

            # Define the system function with constant forcing inside step k
            # TODO: might be possible to change model directly (making it time-variant)
            # TODO: by moving input evaluation and lambda function before the for loop
            # TODO: 'propagate_forced' runs about 30x slower than unforced 'propagate'
            fun = lambda t, x: self.fun(t, x, u_k)

            # Propagate the system one step dt
            sol = solve_ivp(fun, [t, t + self.dt], x_k, method=self.solver)
            if not sol.success:
                error_str = "Solver has encountered an issue:\n{sol_msg}".format(sol_msg=sol.message)
                raise RuntimeError(error_str)

            # Save the simulation results -> x_(k+1)
            new_sim_data[k+1, 1:self.dims+1] = sol.y[:, -1]
            new_sim_data[k+1, self.dims+1:] = u_k

        self.sim_data = np.vstack([self.sim_data, new_sim_data[1:, :]])
        return True


class LorenzSystem(DynamicalSystem):
    def __init__(self, x0, dt=0.005, t0=0,
                 params=None):

        # Lorenz system with default parameters
        fun = lorenz_equation()
        DynamicalSystem.__init__(self, fun, x0, dt, t0, solver='RK45')

class LotkaVolterraSystem(DynamicalSystem):
    def __init__(self, x0, dt=0.005, t0=0):

        fun = lotka_volterra_equation()
        DynamicalSystem.__init__(self, fun, x0, dt, t0, solver='RK45')
