import pandas as pd

from src.utils.containers import DynaFrame as df
import numpy as np

xdata = np.random.rand(5, 3)
dxdata = np.random.rand(5, 3)
udata = np.random.rand(5, 1)
time = np.array([0, 0.1, 0.2, 0.3, 0.4])


dataframe = pd.DataFrame(np.concatenate([time.reshape(-1, 1), xdata], axis=1),
                         columns=('t', 'x1', 'x2', 'x3'))

dataframe_full = pd.DataFrame(np.concatenate([time.reshape(-1, 1), xdata, dxdata, udata], axis=1),
                         columns=('t', 'x1', 'x2', 'x3', 'dx1', 'dx2', 'dx3', 'u1'))

#%%

def test_df_initialization_from_lists():
    dynaframe = df.DynaFrame(xdata, index=time)
    # test index
    expected = time
    real = dynaframe.index
    assert np.all(real == expected)
    # test data
    expected = xdata
    real = dynaframe.values
    assert np.all(real == expected)

def test_df_initialization_from_dataframe_and_reindexing():
    dynaframe = df.DynaFrame(dataframe)
    dynaframe.set_index_from_t()
    expected = time
    real = dynaframe.index
    assert np.all(real == expected)

def test_extraction_of_specified_columns():
    # specified columns -> state vars, state derivative vars or input vars
    dynaframe = df.DynaFrame(dataframe_full)

    expected = ['t', 'x', 'x', 'x', 'd', 'd', 'd', 'u']
    real = dynaframe.var_types
    assert np.all(expected == real)

    expected = np.array(xdata)
    real = np.array(dynaframe.get_state_vars())
    assert np.all(expected == real)

    expected = np.array(dxdata)
    real = np.array(dynaframe.get_state_derivative_vars())
    assert np.all(expected == real)

    expected = np.array(udata)
    real = np.array(dynaframe.get_input_vars())
    assert np.all(expected == real)
