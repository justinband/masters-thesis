import os
from matplotlib.axes import Axes
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy import random
from pathlib import Path


class DataLoader():

    def __init__(self, path=None, seed=None):
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
            self.normalize_data()

            with open(self.data_name, 'wb') as f: # Save data
                pickle.dump(self.data, f)

        self.unique = self.data['normalized'].nunique()

    def _normalize_data(self):
        df = self.data
        col = 'carbon_intensity'
        self.data['normalized'] = np.interp(df[col], (df[col].min(), df[col].max()), (0, 1)) 

    def get_n_samples(self, hourly_window, num_samples):
        """
        Returns: list of samples
        """
        samples = []
        for _ in range(num_samples):
            s = self.sample_range(hourly_window)
            samples.append(s)
        return samples
    
    def split_data(self, train_size):
        """
        train_size must be a percentage 
        """
        data = self.data

        split_idx = int(len(data) * train_size)
        train_df = data.iloc[:split_idx]    # First % for training
        val_df = data.iloc[split_idx:]      # Last (1 - %) for validation
        return train_df, val_df

                
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
    
    def plot_hist(self):
        col = 'carbon_intensity'
        plt.hist(self.data[col],
                 bins=100,
                 density=True,
                 color='royalblue',
                 edgecolor='black',
                 alpha=0.8)
        
        mean = self.data[col].mean()
        std = self.data[col].std()
        plt.axvline(mean, color='darkred', linestyle='solid', linewidth=1.2, label="Mean")
        plt.axvline(mean + std, color='darkgreen', linestyle='solid', linewidth=1.2, label="± SD")
        plt.axvline(mean - std, color='darkgreen', linestyle='solid', linewidth=1.2)

        plt.xlabel(r"Carbon Intensity gCO$_2$eq/kWh (direct)", fontsize=12)
        plt.ylabel("Density", fontsize=12)

        min_date = pd.Timestamp(self.data['date'].min()).strftime('%d/%m/%Y')
        max_date = pd.Timestamp(self.data['date'].max()).strftime('%d/%m/%Y')
        plt.title(f"Distribution of Carbon Intensities from {min_date} to {max_date}")
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot(self):
        plt.figure(figsize=(7.5,7.5))
        plt.plot(self.data['date'], self.data['carbon_intensity']) # date on x-axis and carbon_intensity on y-axis
        ax = plt.gca()

        # Labels
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Show major ticks yearly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as "YYYY-MM"
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))  # Minor ticks every 6 months
        plt.xticks(rotation=45)  # Rotate labels for better readability
        plt.xlabel("Date")
        plt.ylabel("gCO₂eq/kWh")

        # Title
        min_date = pd.Timestamp(self.data['date'].min()).strftime('%d/%m/%Y')
        max_date = pd.Timestamp(self.data['date'].max()).strftime('%d/%m/%Y')
        plt.title(f"Danish Carbon Intensities from {min_date} to {max_date}")

        plt.show()
