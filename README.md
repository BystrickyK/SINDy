# SINDy
Discovering the governing equations of nonlinear systems purely from data using the SINDy method (https://www.pnas.org/content/113/15/3932).
In this project, I've leveraged the SciPy module's ODE solver to simulate dynamical systems with external forcing; the only required input is the system equations and forcing (input) equations.

The resulting simulation data can then be pre-processed for the SINDy method. The 'ProcessedSignal' class takes in state, input and time simulation data, and has the option of adding noise, differentiating (either exact differentiation from the model equations or spectral/forward-diff differentiation from pure data), and convolution filtering.

A library of candidate functions (Theta) is then created from the state and input data, for example all possible polynomials of the given state variables.
The SINDy method is essentially a special case of Ax=b, where A is the library of candidate functions, x the coefficients of the candidate functions and b the state derivatives. The assumption is that the function coefficients - solution x (Xi) - is sparse. 
The solution to Ax=b is therefore given by a sparse-promoting regression - I've tried LASSO, but the SQLS (sequentially thresholded least squares) method from the original paper gave much more accurate results, while being orders of magnitude faster.

After finding the sparse coefficients, an identified model is automatically created as a Python Lambda function, that can then be simulated using the same dynamical system solver to compare the results.
