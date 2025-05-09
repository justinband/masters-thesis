import numpy as np
import matplotlib.pyplot as plt

class CarbonQuantiler():

    def __init__(self, train_data, test_data, alpha):
        self.train_data = train_data
        self.test_data = test_data
        self.alpha = alpha

        self.train_carbon_alpha, self.train_index = self.get_quantile_from_data(self.train_data)
        self.test_carbon_alpha, self.test_index = self.get_quantile_from_data(self.test_data)
        self.test_index += len(self.train_data)
    
    def get_carbon_alphas(self):
        return self.train_carbon_alpha, self.test_carbon_alpha
    
    def get_carbon_alpha_indexes(self):
        return self.train_index, self.test_index

    def get_quantile_from_data(self, data):
        df = data.copy()
        # Sort values
        sorted_df = df.sort_values(by='normalized', ascending=False)
        sorted_values = sorted_df['normalized'].values
        sorted_indexes = np.arange(len(sorted_values))

        try:
            quantile_value = np.quantile(sorted_values, 1/self.alpha) # Find quantile. Can be decimal interpolated.
            pos = np.argmin(np.abs(sorted_values - quantile_value))
            quantile_index = sorted_indexes[pos]

            return quantile_value, quantile_index
        except:
            print(f"Error occurred for quantiling. Values size = {len(sorted_values)}. 1/alpha = {1/self.alpha}")
            raise Exception("ERROR IN QUANTILING")

    

