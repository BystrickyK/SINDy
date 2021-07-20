import pandas as pd
import numpy as np

def cutoff_ends(data, cutoff):

    if isinstance(data, pd.DataFrame):
        return data.iloc[cutoff:-cutoff, :]

    elif isinstance(data, np.ndarray):
        return data[cutoff:-cutoff, :]

