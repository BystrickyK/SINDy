import pandas as pd
import numpy as np

class DynaFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.reinit()

    def reinit(self):
        self.var_types = self.classify_variables()

    def classify_variables(self):
        var_types = []
        for colname in list(self.columns):
            colname = str(colname)
            if ('d' or 'D') in colname:
                var_types.append('d')   # state derivative var
            elif ('x' or 'X') in colname:
                var_types.append('x')   # state var
            elif ('t' or 'time' or 'T') in colname:
                var_types.append('t')   # timestamp
            elif ('u' or 'U' or 'd' or 'D') in colname:
                var_types.append('u')   # input var
            else:
                var_types.append('nan')

        return var_types


    def set_index_from_t(self, index_col_name=None):
        if index_col_name is None:
            index_col = [True if ('t' in vtype) else False
                         for vtype in self.var_types]
            if np.sum(index_col) != 1:
                raise IndexError("There must be exactly one 't' column in the DynaFrame")
            index_col_name = self.columns[index_col][0]
        else:
            if index_col_name not in self.columns:
                raise IndexError("The specified column name is not present in the DynaFrame")

        self.set_index(index_col_name, inplace=True)
        self.index.name = 't'
        self.reinit()
        return True

    def get_dt(self):

        if self.index.name is not 't':
            self.set_index_from_t()

        tdata = self.index.values

        # check that index function is strictly increasing
        if np.all(np.diff(tdata) > 0):
            pass
        else:
            raise IndexError("Time index values must be strictly increasing")

        mean_dt = np.mean(np.diff(tdata))
        mean_dt = mean_dt.round(10)
        if np.all(np.isclose(np.diff(tdata), mean_dt)): # if dt is constant
            dt = mean_dt
            self.dt = dt
        else:
            # TODO: resample
            dt = None
            pass

        return dt

    def type_return_(self, type):
        idx = [True if (type in vartype) else False
                                   for vartype in self.var_types]
        return self.iloc[:, idx]

    def get_state_vars(self):
        return self.type_return_('x')

    def get_state_derivative_vars(self):
        return self.type_return_('d')

    def get_input_vars(self):
        return self.type_return_('u')

def create_df(data, var_label='c'):
    if len(data.shape) == 1:
        dims = 1
    elif len(data.shape) > 1:
        dims = data.shape[1]
    else:
        raise TypeError("Unexpected data shape")

    df = DynaFrame(data)

    if dims > 1:
        var_labels = [var_label + '_' + str(i+1) for i in range(dims)]
        df.columns = var_labels

    return df