import numpy as np
import time
from utils.dynamical_systems import DynamicalSystem
from utils.control_structures import timeout

def calculate_fit(A, x, b):
    # R^2
    error = np.linalg.norm(b - np.dot(A, x))**2 / np.linalg.norm(b)**2
    fit = 1 - error
    return fit, error

def calculate_mse(A, x, b):
    error_squares = np.square(np.dot(A,x) - b)
    mse = np.mean(error_squares)
    return mse

def calculate_rmse(A, x, b):
    mse = calculate_mse(A, x, b)
    rmse = np.sqrt(mse)
    return rmse

# Calculates the Akaike Information Criterion assuming LS estimation
# and normally distributed errors
def calculate_aic(A, x, b):
    mse = calculate_mse(A, x, b)
    n = A.shape[0]
    aic = n*np.log(mse) + 2*np.linalg.norm(x, 0)
    return aic

# def calculate_model_validation(models, sim_data, y_real):
#
#     dx3model = sim_data.apply(model['eqn_lambda'], axis=1)