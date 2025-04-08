import numpy as np

def calculate_regret(carbon, opt_carbon):
    decimals = 5
    # carbon = np.round(carbon, decimals)
    # opt_carbon = np.round(opt_carbon, decimals)
    return np.round(carbon - opt_carbon, decimals)
