import pandas as pd
import numpy as np

def read_csv(*args, **kwargs):
    df = pd.read_csv(*args, **kwargs)
    df.