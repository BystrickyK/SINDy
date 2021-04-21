import numpy as np
import time
from utils.dynamical_systems import DynamicalSystem
from utils.control_structures import timeout
from utils.signal_processing import StateSignal, ForcingSignal

def calculate_fit(A, x, b):
    # R^2
    error = np.linalg.norm(b - np.dot(A, x))**2 / np.linalg.norm(b)**2
    fit = 1 - error
    return fit, error

def calculate_rmse(A, x, b):
    A = A.values
    x = np.array(x)
    error_squares = np.square(np.dot(A,x) - b)
    error = np.sqrt(error_squares)
    rmse = np.mean(error)
    return rmse

