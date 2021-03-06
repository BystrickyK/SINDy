import numpy as np
import pandas as pd
from utils.function_libraries import sum_library

data = np.ones((10,3)) * [1,2,3]
df = pd.DataFrame(data)
df.columns = ['X1', 'X2', 'X3']

lib = sum_library(df, (-2, -1, 0, 1, 2))