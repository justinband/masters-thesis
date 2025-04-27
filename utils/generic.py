import numpy as np

def calculate_regret(carbon, opt_carbon, decimals=5):
    return np.round(carbon - opt_carbon, decimals)

def calculate_diff(old_val, new_val):
    diff = ((new_val - old_val) / old_val) * 100
    return np.round(diff, 3)
