import numpy as np

def finite_diff(x, dt, direction='forward'):
    """
    x (DataFrame): State measurements
    dt (Float): Time step size
    """
    if direction == 'forward':
        dxdt = (np.diff(x, axis=0)) / dt  # last value is missing
        dxdt = np.vstack((dxdt, dxdt[-1, :]))
        return dxdt
    elif direction == 'backward':
        x = np.flip(x.values, axis=0)
        dxdt = (-np.diff(x, axis=0)) / dt
        dxdt = np.flip(dxdt, axis=0)  # first value is missing
        dxdt = np.vstack((dxdt[0, :], dxdt))
        return dxdt
