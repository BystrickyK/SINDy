import re

# cuts off a specified number of rows from both sides of a pandas DataFrame
def cutoff(x, idx_cutoff):
    x = x.iloc[idx_cutoff:-idx_cutoff]
    return x

def parse_function_strings(theta_cols):
    fun_strings = []
    for fun in theta_cols:
        print(fun)
        splits = fun.split('*')
        newvars = []
        if splits[0] == '1':
            funstr = rf"$ {splits[0]} $"
            fun_strings.append(funstr)
            continue
        for var in splits:
            idx = re.search('\d', var) # find number in var
            idx = str(int(idx.group(0))+1)
            varname = re.search('^\w', var) # pick first char in var
            varname = varname.group(0)
            newvars.append(rf'{varname}_{idx} ')
        funstr = rf"$ {''.join(newvars)} $"
        fun_strings.append(funstr)
    return fun_strings

def latexify(theta_cols):
    col_strings = []
    for col in theta_cols:
        col_strings.append(rf'$ {col} $')
    return col_strings

def d_to_dot(theta_cols):
    col_strings = []
    for col in theta_cols:
        col = col.replace('dx', '\\dot{x}')
        col_strings.append(col)
    return col_strings

def parse_function_str_add_dots(theta_cols):
    new_vars = []
    for var in theta_cols:
        result = re.search('\$\ (.*)\ \ \$', var)
        result = result.group(1)
        idx = re.search('\d', result)  # find number in var
        idx = str(int(idx.group(0)))
        varname = re.search('^\w', result)  # pick first char in var
        varname = varname.group(0)
        var_new = rf'$ {{\}}dot{{ {varname} }}_{idx} $'.replace('{\}', '\\')
        new_vars.append(var_new)
    return new_vars


