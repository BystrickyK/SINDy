import sympy as sp
import numpy as np
import pandas as pd

class LibraryCreator:
    def __init__(self, target, function_strings, data):
        self.target = target
        self.function_strings = function_strings
        self.data = data

        self.lambdify_vars(data.columns)

        self.lambdify_function_strings()

    def lambdify_vars(self, vars):
        symvars = []
        for var in vars:
            var = sp.parse_expr(var)
            symvars.append(var)
        self.symvars = symvars

    def lambdify_function_strings(self):
        fun_symeqns = []
        fun_lambdas = []
        for str in self.function_strings:
            symeqn = sp.parse_expr(str)
            lamb = sp.lambdify(self.symvars, symeqn)

            fun_symeqns.append(symeqn)
            fun_lambdas.append(lamb)

        self.fun_symeqns = fun_symeqns
        self.fun_lambdas = fun_lambdas

    def create_library(self):
        data = np.empty([len(self.data), len(self.fun_lambdas)])
        for col, fun_str in enumerate(self.function_strings):
            fun = self.fun_lambdas[col]
            rowfun = lambda row: fun(*row)
            data[:, col] = self.data.apply(rowfun, axis=1, raw=True).values
        df = pd.DataFrame(data)
        df.columns = self.function_strings
        return df
