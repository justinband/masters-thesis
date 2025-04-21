import numpy as np

def calculate_regret(carbon, opt_carbon, decimals=5):
    return np.round(carbon - opt_carbon, decimals)
