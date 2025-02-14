import os
from matplotlib.axes import Axes
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
        
    def find_optimal_intensities(self, range_df, job_size, find_subrange=False, show_plots=False):
        """
        Given a range, this function finds optimal intensities relating to the job size.
        In essence, these are the best carbon intensities in hindsight. It finds and returns the
        dates with the minimum carbon intensities. This corresponds with scheduling a job and being
        able to run/pause.

        This function also allows for finding a subrange, rather than disjoint data. This exists
        as a "low carbon range" might be more practical and important. It corresponds with
        scheduling a job and not being allowed to pause it until it's complete.

        Lastly, plots can be created for these ranges for better comparison.
        """
        # best possible run/pause with hindsight
        opt_run_pause_df = range_df.sort_values(by='carbon_intensity')[0:job_size]
        opt_run_pause_ci = opt_run_pause_df['carbon_intensity'].sum()
        
        # best possible subrange with only run in hindsight
        opt_sub_df = None
        opt_sub_ci = 0
        min_mean = float('inf')

        if find_subrange:
            # Init sum for first subrange
            subrange_sum = range_df.iloc[:job_size]['carbon_intensity'].sum()
            for curr_ix in range(range_df.shape[0] - job_size + 1):
                subrange_mean = subrange_sum / job_size

                if subrange_mean < min_mean:
                    min_mean = subrange_mean
                    opt_sub_df = range_df.iloc[curr_ix:(curr_ix + job_size)]
                    opt_sub_ci = subrange_sum

                if curr_ix + job_size < range_df.shape[0]:
                    subrange_sum -= range_df.iloc[curr_ix]['carbon_intensity']
                    subrange_sum += range_df.iloc[curr_ix + job_size]['carbon_intensity']


        if show_plots:
            if find_subrange:
                _, ax = plt.subplots(1, 2, figsize=(10, 5))
                # Run/Pause
                rp_xs = opt_run_pause_df['date'].to_numpy()
                rp_ys = opt_run_pause_df['carbon_intensity'].to_numpy()
                ax[0].plot(range_df['date'], range_df['carbon_intensity'], 'o-', label='Carbon Intensity')
                ax[0].plot(rp_xs, rp_ys, 'ro', label='Highlighted Points') 
                ax[0].set_ylabel(r"gCO$_2$eq/kWh")
                ax[0].set_xlabel("Date")
                ax[0].set_title(f"[Optimal] Carbon Intensity: {opt_run_pause_ci:.2f}")
                ax[0].legend()

                # Subrange
                opt_xs = opt_sub_df['date'].to_numpy()
                opt_ys = opt_sub_df['carbon_intensity'].to_numpy()
                ax[1].plot(range_df['date'], range_df['carbon_intensity'], 'o-', label='Carbon Intensity')
                ax[1].plot(opt_xs, opt_ys, 'ro', label='Highlighted Points') 
                ax[1].set_ylabel(r"gCO$_2$eq/kWh")
                ax[1].set_xlabel("Date")
                ax[1].set_title(f"[Optimal Subrange] Carbon Intensity: {opt_sub_ci:.2f}")
                ax[1].legend()

                plt.show()
            else:
                # Run/Pause
                rp_xs = opt_run_pause_df['date'].to_numpy()
                rp_ys = opt_run_pause_df['carbon_intensity'].to_numpy()
                plt.plot(range_df['date'], range_df['carbon_intensity'], 'o-', label='Carbon Intensity')
                plt.plot(rp_xs, rp_ys, 'ro', label='Highlighted Points') 
                plt.ylabel(r"gCO$_2$eq/kWh")
                plt.xlabel("Date")
                plt.title(f"[Optimal] Carbon Intensity: {opt_run_pause_ci:.2f}")
                plt.legend()
                plt.show()

        return opt_run_pause_df, opt_run_pause_ci, opt_sub_df, opt_run_pause_ci


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
