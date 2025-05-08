import numpy as np

class CarbonQuantiler():

    def __init__(self, train_data, test_data, alpha):
        self.train_data = train_data
        self.test_data = test_data
        self.alpha = alpha

        self.train_carbon_alpha, self.train_index = self._get_quantile_from_data(self.train_data)
        self.test_carbon_alpha, self.test_index = self._get_quantile_from_data(self.test_data)

        self.test_index += len(self.train_data)
    
    def get_carbon_alphas(self):
        return self.train_carbon_alpha, self.test_carbon_alpha
    
    def get_carbon_alpha_indexes(self):
        return self.train_index, self.test_index

    def _get_quantile_from_data(self, data):
        df = data.copy()
        sorted_values = df.sort_values(by='normalized', ascending=False)['normalized'].values
        sorted_indexes = df.sort_values(by='normalized', ascending=False)['normalized'].index
        index_result = np.quantile(sorted_indexes, 1/self.alpha)
        value_result = np.quantile(sorted_values, 1/self.alpha)
        return value_result, index_result
