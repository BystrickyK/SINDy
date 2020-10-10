
# cuts off a specified number of rows from both sides of a pandas DataFrame
def cutoff(x, idx_cutoff):
    x = x.iloc[idx_cutoff:-idx_cutoff]
    return x
