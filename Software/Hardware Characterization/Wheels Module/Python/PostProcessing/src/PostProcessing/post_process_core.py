import os
import os.path as pth

import statsmodels as sm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#############################################################################################################################
################################################## MotorData Class ##########################################################
#############################################################################################################################
class MotorData:

    def __init__(self, filename:str, rolling_window_size = 10, filepath = pth.join(pth.dirname(__file__), "data")):
        
        self.filename = filename
        #Read in data, get rid of rows with duplicate time stamps
        raw_data = pd.read_csv(pth.join(filepath, filename))
        unique_times = raw_data['Time'].drop_duplicates()
        data_unique_times = raw_data.copy().iloc[unique_times.index]

        #Get rid of spurious pwm values, reset index
        self.data = self.clean_series(data_unique_times, 'PWM', rolling_window_size)
        self.data.index = range(len(self.data))

    def clean_series(self, data, series_name:str, window_size):

        #pulled code for generating rolling average and filtering for it from Google AI result
        rolling_mean = data[series_name].rolling(window = window_size, center = True).mean()
        rolling_std = data[series_name].rolling(window = window_size, center= True).std()

        lower_bound = rolling_mean - 2 * rolling_std
        upper_bound = rolling_mean + 2 * rolling_std

        is_not_outlier = (data[series_name] >= lower_bound) & (data[series_name] <= upper_bound)

        return data[is_not_outlier]

    def plot(self, x_axis:str, y_axis:str, fig = None, ax = None, line_format:str = '-'):
        """
        Generates simple line and scatter plots for specified series on a specified figure
        and axis. Assumes a simple line plot unless told otherwise.
        """

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols = 1)

        ax.plot(self.data[x_axis], self.data[y_axis], line_format)

        #By default, have the x and y axis labels as well as the title be the variables. Can overwrite
        #with ax.set_xlabel, ax.set_ylabel, ax.set_title after the self.plot call
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title('{} vs {}'.format(y_axis, x_axis))

    def get_step_windows(self, motor_commands:list) -> dict:
        """
        Obtain the windows for which step repsonse values are obtained on the dataset for each command and run.

        Does so through simple logical indexing, and time shifts the data so it starts at zero for each command and run pair.
        """
        runs = self.count_runs()
        sr_values = {run:{command:pd.DataFrame() for command in motor_commands} for run in runs} #Store all of the steady-state values for each command

        for run in runs:
            for command in motor_commands:
                slice = self.data[(self.data['PWM'] == command) & (self.data['Run'] == run)].copy()

                #Need to eliminate chunks that are spurious
                slice = self.clean_series(slice, 'Time', window_size = 10)

                startTime = slice['Time'].min()
                slice['Time'] = slice['Time'] - startTime
                slice.index = range(len(slice))
                sr_values[run][command] = slice

        return sr_values
    
    def estimate_ss(self, motor_commands:list) -> dict:
        """
        Based on the step windows obtained, retrieve steady state speed and current estimates for each command and run number.

        Does this under the assumption that we've hit both within the last 5ish seconds of each run. 
        """

        sr_values = self.get_step_windows(motor_commands)
        runs = sr_values.keys()
        ss_values = {run:{command:{"Speed":0.0, "Current":0.0, "Time Constant":0.0, "Var_Theta":0.0, "Var_Omega":0.0} for command in motor_commands} for run in runs} #Store all of the steady-state values for each command

        for run in runs:
            for command in motor_commands:
                #Grab last 4 seconds (ish, last second gets reverse EMF on current readings)
                criteria = (sr_values[run][command]['Time'] > 10) & (sr_values[run][command]['Time'] < 14)
                last_four_seconds = sr_values[run][command][criteria].copy()
                
                #Linear regression to get speed estimate
                X = last_four_seconds['Time'].values.reshape(-1,1)
                y = last_four_seconds['Position'].values
                try:
                    speed_ss_estimates = np.diff(y) / np.diff(X.reshape(y.shape))
                    ss_values[run][command]["Var_Omega"] = np.var(speed_ss_estimates)
                    speed_curve = sm.OLS(y,X).fit()
                    ss_values[run][command]["Speed"] = speed_curve.params[0]
                    ss_values[run][command]["Var_Theta"] = speed_curve.mse_resid
                    omega = ss_values[run][command]["Speed"]
                    tau_series = (last_four_seconds['Time'] * omega - last_four_seconds['Position']) / omega
                    ss_values[run][command]["Time Constant"] = tau_series.mean()
                except ValueError:
                    #Regression seems to be randomly failing due to there being nonsense ss values
                    #Full bug seemed to be the run randomly updating before the command did, and all the fun
                    #associated with that
                    print("Regression failed on Run {} Command {}".format(run, command))

                #SS current estimate
                ss_values[run][command]["Current"] = last_four_seconds['Current'].mean()

        return ss_values
    

    def compute_voltage(self, PWM, i_ss):

        nominal_voltage = 12.24 #Volts
        shunt_resistance = .2609938 #Ohms

        voltage = (PWM / 255.0) * nominal_voltage - i_ss * shunt_resistance

        return voltage
    
    def count_runs(self, max_runs = 5):

        runs = list(self.data['Run'].drop_duplicates())
        if 0 in runs:
            runs.remove(0) #Run 0 is sometimes a thing apparently
        if max_runs+1 in runs:
            runs.remove(max_runs+1)

        return runs

#############################################################################################################################
############################################# Helper/Packing Functions ######################################################
#############################################################################################################################    

def ExtractMotorLabels(filename) -> tuple:
    """
    Extracts the motor label and block ID from the filename.

    Assumes a specific naming convention
    """

    split_filename = filename.split('_',7)

    motor_label = split_filename[3][-1]
    block_ID = int(split_filename[5])

    return motor_label, block_ID
    

def ParseMotorData(motor_data_list:list[MotorData], commands:list) -> pd.DataFrame:

    #build dataframe
    motor_data_dict = {"Motor":[], "Block":[], "Run":[], 
                       "Command":[], "Speed":[], "Current":[], "Time Constant":[], "Voltage":[], 
                       "Var_Theta":[], "Var_Omega":[]}

    for motor_data in motor_data_list:

        #Extract motor label and block number
        motor_label, block = ExtractMotorLabels(motor_data.filename)

        #Extract the steady-state values
        ss_value_dict = motor_data.estimate_ss(commands)

        for run in ss_value_dict.keys():
            for command in commands:

                speed = ss_value_dict[run][command]["Speed"]
                current = ss_value_dict[run][command]["Current"]
                time_constant = ss_value_dict[run][command]["Time Constant"]
                theta_var = ss_value_dict[run][command]["Var_Theta"]
                omega_var = ss_value_dict[run][command]["Var_Omega"]
                voltage = motor_data.compute_voltage(command, current)

                motor_data_dict["Block"].append(block)
                motor_data_dict["Motor"].append(motor_label)
                motor_data_dict["Run"].append(run)
                motor_data_dict["Command"].append(command)
                motor_data_dict["Speed"].append(speed)
                motor_data_dict["Current"].append(current)
                motor_data_dict["Time Constant"].append(time_constant)
                motor_data_dict["Voltage"].append(voltage)
                motor_data_dict["Var_Theta"].append(theta_var)
                motor_data_dict["Var_Omega"].append(omega_var)

    return pd.DataFrame(motor_data_dict)

def ReshapeMotorData(motor_data_list:list[MotorData], motor_labels:list[str], block_labels:list[int]) -> dict[str,dict[int,MotorData]]:

    reshaped_dict = {motor:{block:None for block in block_labels} for motor in motor_labels}

    for motor_data in motor_data_list:

        motor_label, block_label = ExtractMotorLabels(motor_data.filename)
        reshaped_dict[motor_label][block_label] = motor_data

    return reshaped_dict

def GetTimesAndVoltages(data:MotorData, sr_values:pd.DataFrame):

    times = sr_values['Time'].values
    currents = sr_values['Current'].values
    pwms = sr_values['PWM'].values
    voltages = data.compute_voltage(pwms, currents)

    return times, voltages