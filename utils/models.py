import numpy as np



def create_SINDy_model(candidate_functions, ksi, thresh=0.01):
    cand_fun = theta.columns
    ksiT = ksi.T

    # Full system function string
    system_str = ''
    # Build function strings for each state function out of xi coefficients and candidate function labels
    state_fun_strings = []
    for state_fun_idx in range(ksiT.shape[1]):
        system_str += "State function x{}_dot\n".format(state_fun_idx)
        state_fun_str = ''
        for cand_fun_str, cand_fun_coeff in zip(candidate_functions, ksiT[:, state_fun_idx]):
            if np.abs(cand_fun_coeff) > thresh:
                cand_str = "{c:0.5f} * {fun} + ".format(c=cand_fun_coeff, fun=cand_fun_str) # rounds to 5 decimal places
                state_fun_str += cand_str
                system_str += "\t{}\n".format(cand_str)
        state_fun_str = state_fun_str[:-3]  # cut off last 3 characters (the plus sign and two spaces)
        state_fun_strings.append(state_fun_str)
        system_str = system_str[:-3] + '\n\n'

    # Combine the state function strings into lambda output form
    lambda_str = 'lambda x, u: ['
    for state_fun_str in state_fun_strings:
        lambda_str += state_fun_str + ', '
    lambda_str = lambda_str[:-2] + ']'  # cut off last two characters and add ']'

    identified_model = eval(lambda_str)  # SINDYc identified model
    return identified_model, system_str
