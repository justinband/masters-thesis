import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy import random

class DataLoader():
    def __init__(self):
        self.data_name = 'dk_energy_data.pkl'

        if os.path.isfile(self.data_name): # Load data from pkl file
            with open(self.data_name, 'rb') as f:
                self.data = pickle.load(f)
        else: # Load data from source
            arr = []
            for fn in [fn for fn in os.listdir('./') if '.csv' in fn]:
                arr.append(pd.read_csv(fn, sep=',')[["Datetime (UTC)", "Carbon Intensity gCO₂eq/kWh (direct)"]])

            df = pd.concat(arr)
            df = df.rename(columns={"Datetime (UTC)": "date", "Carbon Intensity gCO₂eq/kWh (direct)": "carbon_intensity"})
            df = df.astype({'date': 'datetime64[ns]'})
            self.data = df.sort_values(by='date').reset_index(drop=True) # Sort by date
            self.normalize_data()

            with open(self.data_name, 'wb') as f: # Save data
                pickle.dump(self.data, f)

    def normalize_data(self):
        df = self.data
        col = 'carbon_intensity'
        self.data['normalized'] = np.interp(df[col], (df[col].min(), df[col].max()), (0, 1)) 
                
    def sample_range(self, hourly_window, seed=None):
        if seed:
            random.seed(seed)

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
            print(x)
            diff = end_ix - nrows.stop
            first = self.data.iloc[start_ix:nrows.stop, :]
            wrap = self.data.iloc[0:diff, :]
            df = pd.concat([first, wrap])
            return df
        else: # Data fits
            return self.data.iloc[start_ix:end_ix, :]
        
    def get_optimal_subrange(self, range_df, job_size, use_subrange=True):
        """
        Given a range data, this function finds the subrange with the minimum average
        values.

        For carbon intensity data, this means that means to run without stopping.
        """
        
        if use_subrange:
            # best possible subrange with only run in hindsight
            opt_sub_df = None
            min_mean = float('inf')

            # Init sum for first subrange
            subrange_sum = range_df.iloc[:job_size]['carbon_intensity'].sum()
            for curr_ix in range(range_df.shape[0] - job_size + 1):
                subrange_mean = subrange_sum / job_size

                if subrange_mean < min_mean:
                    opt_sub_df = range_df.iloc[curr_ix:(curr_ix + job_size)]
                    min_mean = subrange_mean

                if curr_ix + job_size < range_df.shape[0]:
                    subrange_sum -= range_df.iloc[curr_ix]['carbon_intensity']
                    subrange_sum += range_df.iloc[curr_ix + job_size]['carbon_intensity']
        else:
            # best possible run/pause with hindsight
            opt_sub_df = range_df.sort_values(by='carbon_intensity')[0:job_size]


        plt.plot(range_df['date'], range_df['carbon_intensity'], label='Carbon Intensity')
        for val in opt_sub_df['date']:
            plt.axvline(val, color='darkred')
        plt.legend()
        plt.show()        


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

x = DataLoader()
range_sample = x.sample_range(hourly_window=48, seed=123)
x.get_optimal_subrange(range_df=range_sample, job_size=10)

