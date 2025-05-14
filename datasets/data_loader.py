import os
from matplotlib.axes import Axes
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from numpy import random
from pathlib import Path
from .carbon_quantiler import CarbonQuantiler

class DataLoader():

    def __init__(self, train_size, alpha, path=None, seed=None):
        DATA_DIR = Path(__file__).parent
        if path:
            DATA_DIR = path
        self.data_name = os.path.join(DATA_DIR, 'dk_energy_data.pkl')

        if seed:
            random.seed(seed)
        
        if os.path.isfile(self.data_name): # Load data from pkl file
            with open(self.data_name, 'rb') as f:
                self.data = pickle.load(f)
        else: # Load data from source
            arr = []
            filenames = [os.path.join(DATA_DIR, fn) for fn in os.listdir(path) if '.csv' in fn]
            for fn in filenames:
                print(fn)
                arr.append(pd.read_csv(fn, sep=',')[["Datetime (UTC)", "Carbon Intensity gCO₂eq/kWh (direct)"]])

            df = pd.concat(arr)
            df = df.rename(columns={"Datetime (UTC)": "date", "Carbon Intensity gCO₂eq/kWh (direct)": "carbon_intensity"})
            df = df.astype({'date': 'datetime64[ns]'})
            self.data = df.sort_values(by='date').reset_index(drop=True) # Sort by date
            self._normalize_data()

            with open(self.data_name, 'wb') as f: # Save data
                pickle.dump(self.data, f)

        self.unique = self.data['normalized'].nunique()
        self.split_idx = 0
        self.train_size = train_size
        self.train_data, self.test_data = self._train_test_split(self.data, train_size)

        self.alpha = alpha
        self.carbon_quantiler = CarbonQuantiler(self.train_data, self.test_data, self.alpha)
        self.train_ca, self.train_ca_index = self.carbon_quantiler.get_quantile_from_data(self.train_data)
        self.test_ca, self.test_ca_index = self.carbon_quantiler.get_quantile_from_data(self.test_data)
        self.test_ca_index += len(self.train_data)
        self.plot_carbon_alphas()

    def get_ca_from_idx(self, idx, is_train):
        # Split data for before index
        data = self.train_data[:idx] if is_train else self.test_data[:idx]
        value, index = self.carbon_quantiler.get_quantile_from_data(data)
        return value

    def _train_test_split(self, data, train_size):
        self.split_idx = int(len(data) * train_size)
        train_df = data.iloc[:self.split_idx].reset_index(drop=True)
        test_df = data.iloc[self.split_idx:].reset_index(drop=True)
        return train_df, test_df
    
    def _normalize_data(self):
        df = self.data
        col = 'carbon_intensity'
        self.data['normalized'] = np.interp(df[col], (df[col].min(), df[col].max()), (0, 1)) 
        self.data['normalized'] = np.round(self.data['normalized'], 5) # Analysis in data_analysis.ipynb determined 5 decimals is okay

    def get_data_split(self):
        return self.train_data, self.test_data

    def get_n_samples(self, hourly_window, num_samples):
        """
        Returns: list of samples
        """
        samples = []
        for _ in range(num_samples):
            s = self.sample_range(hourly_window)
            samples.append(s)
        return samples

                
    def sample_range(self, hourly_window):
        nrows = range(self.data.shape[0])
        start_ix = random.randint(nrows.start, nrows.stop)
        end_ix = start_ix + hourly_window

        # Two Options for handling end_ix > nrows.stop
        #   1. Generate new indexes
        #       May never use recent data, or last data
        #   2. Loop back on data
        #       Could introduce large jumps in data
        # We pursue (2)

        if end_ix > nrows.stop: # Gone past available data
            diff = end_ix - nrows.stop
            first = self.data.iloc[start_ix:nrows.stop, :]
            wrap = self.data.iloc[0:diff, :]
            df = pd.concat([first, wrap])
            return df
        else: # Data fits
            return self.data.iloc[start_ix:end_ix, :]
        
    def _get_optimal_intensities(self, df, job_size, get_subrange=False):
        opt_df = df.nsmallest(job_size, 'carbon_intensity')
        opt_total = opt_df['carbon_intensity'].sum()

        if get_subrange:
            min_intensity = float('inf')
            subrange_df, subrange_total = None, None
        
            rolling_sum = df.iloc[:job_size]['carbon_intensity'].sum()
            for i in range(len(df) - job_size + 1):
                avg_intensity = rolling_sum / job_size

                if avg_intensity < min_intensity:
                    min_intensity = avg_intensity
                    subrange_df = df.iloc[i:i + job_size]
                    subrange_total = rolling_sum
                
                if i + job_size < len(df):
                    rolling_sum -= df.iloc[i]['carbon_intensity']
                    rolling_sum += df.iloc[i + job_size]['carbon_intensity']

            return opt_df, opt_total, subrange_df, subrange_total
        
        return opt_df, opt_total, None, None

    def _plot_carbon_data(self, ax, df, selected_df, title):
        ax.plot(df['date'], df['carbon_intensity'], 'o-', label='Carbon Intensity')
        ax.plot(selected_df['date'], selected_df['carbon_intensity'], 'ro', label='Optimal Points')
        ax.set_ylabel(r"gCO$_2$eq/kWh")
        ax.set_xlabel("Date")
        ax.set_title(title)

        # TODO: Fix this to be nicer
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%d-%m-%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

    def _visualize_optimal_carbon_data(self, df, opt_df, opt_ci, sub_df=None, sub_ci=None):
        if sub_df is not None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            self._plot_carbon_data(ax[0], df, opt_df, f"[Optimal] Carbon Intensity: {opt_ci:.2f}")
            self._plot_carbon_data(ax[1], df, sub_df, f"[Optimal Subrange] Carbon Intensity: {sub_ci:.2f}")
        else:
            fix, ax = plt.subplots(figsize=(7, 5))
            self._plot_carbon_data(ax, df, opt_df, f"[Optimal] Carbon Intensity: {opt_ci:.2f}")
        
        plt.show()

    def find_and_plot_optimal_ci(self, df, job_size, find_subrange=False, show_plots=False):
        opt_df, opt_ci, sub_df, sub_ci = self._get_optimal_intensities(df, job_size, get_subrange=find_subrange)
        if show_plots:
            self._visualize_optimal_carbon_data(df, opt_df, opt_ci, sub_df, sub_ci)
        return opt_df, opt_ci, sub_df, sub_ci

    def plot_quantile_ranges(self, alpha):
        data = self.data.copy()
        data = data.sort_values(by='normalized', ascending=False)['normalized'].values
        plt.plot(data)

        quantile = 1/alpha
        quantile_value = np.quantile(data, quantile)

        indexes = np.where(data <= quantile_value)[0]
        min_index = indexes[0]
        max_index = indexes[-1]

        plt.axvline(min_index, color='red', label=f'{(quantile * 100):.0f}% quantile', alpha=0.3)
        plt.axvspan(min_index, max_index, ymin=0, ymax=1, color='skyblue', alpha=0.3)
        
        plt.axhline(quantile_value, color='red', label=f'{quantile_value}')

        plt.ylabel("Normalzed Carbon Intensity")
        plt.title(f"{(quantile * 100):.0f}% Quantile of Sorted Carbon Intensities")
        plt.legend()
        plt.show()

    def plot_carbon_alphas(self):
        energy_type = 'normalized'

        sns.set_theme(style='darkgrid')

        plt.figure(figsize=(10, 6))
        data = self.data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date').sort_index()
        sns.lineplot(x=data[energy_type].index, y=data[energy_type].values, linewidth=1.2, alpha=0.5)

        # Descending Train and Test data
        train_sorted = self.train_data.sort_values(by='normalized', ascending=False)['normalized']
        test_sorted = self.test_data.sort_values(by='normalized', ascending=False)['normalized']
        plt.plot(pd.date_range(data.index[0],data.index[self.split_idx-1], freq='h'),
                 train_sorted,
                 label='Descending train data',
                 linewidth=2)
        plt.plot(pd.date_range(data.index[self.split_idx],data.index[-1], freq='h'),
                 test_sorted,
                 label='Descending test data',
                 linewidth=2)

        plt.axvline(data.index[self.split_idx], label='Train/Test Split', color='red')

        # plt.axhline(y=self.train_ca, xmin=0, xmax=self.split_idx/len(self.data),color='purple')
        # plt.axvline(data.index[self.train_ca_index], label=r'Train $c_{1/\beta}$', linestyle='-.', color='purple')
        plt.scatter(x=data.index[self.train_ca_index],
                    y=self.train_ca,
                    label=fr"Train $c_{{1/\beta}} = {np.round(self.train_ca, 3)}$",
                    s=100,
                    zorder=5,
                    color='darkorange')
        sns.lineplot(
            x=pd.date_range(data.index[0], data.index[self.train_ca_index], freq='h'),
            y=self.train_ca,
            color='grey',
            linestyle='--',
            alpha=0.5
        )

        # plt.axhline(y=self.test_ca, color='darkblue')
        # plt.axvline(data.index[self.test_ca_index], label=r'Test $c_{1/\beta}$', linestyle='-.', color='darkblue')
        plt.scatter(x=data.index[self.test_ca_index],
                    y=self.test_ca,
                    label=fr'Test $c_{{1/\beta}} = {np.round(self.test_ca, 3)}$',
                    s=100,
                    zorder=5,
                    color='darkgreen')
        sns.lineplot(
            x=pd.date_range(data.index[0], data.index[self.test_ca_index], freq='h'),
            y=self.test_ca,
            color='grey',
            linestyle='--',
            alpha=0.5
        )

        

        plt.ylabel("Normalized Carbon Intensity", labelpad=10)
        plt.xlabel("Date", labelpad=10)
        plt.title(fr"Carbon quantiles for {int(self.train_size * 100)}% Train/Test Split with $\beta = {self.alpha}$")
        plt.legend()
        sns.despine()
        plt.tight_layout()
        plt.savefig('./figures/dataset/all_caron_alpha.png', dpi=300)
        plt.show()