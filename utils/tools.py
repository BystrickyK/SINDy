import re

# cuts off a specified number of rows from both sides of a pandas DataFrame
def cutoff(x, idx_cutoff):
    x = x.iloc[idx_cutoff:-idx_cutoff]
    return x

def parse_function_strings(theta):
    fun_strings = []
    for fun in theta:
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

def parse_function_str_add_dots(vars):
    new_vars = []
    for var in vars:
        result = re.search('\$\ (.*)\ \ \$', var)
        result = result.group(1)
        idx = re.search('\d', result)  # find number in var
        idx = str(int(idx.group(0)))
        varname = re.search('^\w', result)  # pick first char in var
        varname = varname.group(0)
        var_new = rf'$ {{\}}dot{{ {varname} }}_{idx} $'.replace('{\}', '\\')
        new_vars.append(var_new)
    return new_vars


