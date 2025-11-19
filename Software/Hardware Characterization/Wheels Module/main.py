"""
MotorCharacterize_SpeedBoost
"""

#Standard library imports: basic functionality
import os
import os.path as pth
import pandas as pd
import numpy as np

#Serial communication and post-processing
from SerialStreamParser import GatherAndStore
from PostProcessing import MotorData, ParseMotorData, ReshapeMotorData, GetTimesAndVoltages

#generating unique filename for date and time
from datetime import datetime

#MCMC code and simulations
from MCMC_Simulator import HierarchicalModel, InitializeSimulations, RunBlocks, RunCommandForParams, PackOutputList
from StateModels import MotorModel

#Kalman Filter, MLE fitting
from Filters import EWMA, StateObserver

#Plotting, as well as loading in trace data from MCMC
import matplotlib.pyplot as plt
import arviz as az 

#Set the random seed (done after bayesian and monte carlo fitting procedure)
np.random.seed(0)
    

#############################################################################################################################
############################################# Helper/Packing Functions ######################################################
#############################################################################################################################   


def WriteParamsToFile(param_dict:dict[str,float], file) -> None:
    for param in param_dict.keys():
        file.write("{} = {}\n".format(param, param_dict[param]))

    file.write("\n")

def OpenParamsFromFile(filename:str, param_keys:list[str]) -> tuple[dict[str, float], list[dict[str,float]]]:

    params_list_motors = []
    with open(filename, 'r') as file:

        chunks = file.read().split('\n\n')
        for chunk in chunks:
            lines = chunk.split('\n')

            params_dict = {param:0 for param in param_keys}

            for line in lines[1:]:

                key_value_separation = line.split(' = ')
                try:
                    params_dict[key_value_separation[0]] = float(key_value_separation[1].strip())
                except KeyError:
                    raise KeyError("Mismatch between expected keys and zeroth value of key_value_separation")

            if 'Motor' in lines[0]:
                params_list_motors.append(params_dict)
            
            elif 'Global' in lines[0]:
                global_params_dict = params_dict.copy()
            else:
                pass
    
    return global_params_dict, params_list_motors
                
            
#############################################################################################################################
######################################################## Main ###############################################################
#############################################################################################################################
def main():

    #Serial communication variables
    COM_PORT = "COM3"
    BAUD_RATE = 230400

    time_cutoff = 800 #Record for 800 seconds

    #File naming information, timestamps each file it creates to prevent accidental overwrites
    motor = 'C'
    block_number = 1
    date_info = datetime.now()
    mmddyyyy = "{}{}{}_{}{}".format(date_info.month, date_info.day, date_info.year,date_info.hour, date_info.minute)

    #First line is for general fitting data files, second for filter testing
    data_filename = "motor_data_romeo_Motor{}_Block_{}_{}".format(motor, block_number, mmddyyyy)
    # data_filename = "motor_data_romeo_Motor{}_{}".format(motor, mmddyyyy)

    #First line is for general fitting data files, second for filter testing
    data_folder = pth.join(pth.dirname(__file__), "data/RCBD_Trials")
    # data_folder = pth.join(pth.dirname(__file__), "data/Disturbance_Tests")

    #lists of commands, to be used in plotting and processing steps
    multi_speed_commands = [31, 63, 95, 127, 159, 191, 223, 255]

    #Flags that control what the program is doing: collecting data, providing early visualizations, or preprocessing data
    collecting_data = False
    plot_raw_data = False
    preprocess_data = False

    #Model identification
    mcmc_ready = False
    compute_nominal_params = False

    #Filter tuning 
    tune_filter = True

    if collecting_data:

        GatherAndStore(COM_PORT, BAUD_RATE, time_cutoff, data_filename)
        
    if plot_raw_data:

        multi_speed_files = os.listdir(data_folder)
        MultiData = [MotorData(cell, filepath=data_folder) for cell in multi_speed_files]

        #Plot time vs PWM and Run number for data to check if we are cycling through commands correctly
        MultiData[0].plot('Time','Position')
        plt.show()

    if preprocess_data:

        multi_speed_files = os.listdir(data_folder)
        MultiData = [MotorData(cell, filepath=data_folder) for cell in multi_speed_files]
        
        aggregated_data = ParseMotorData(MultiData, multi_speed_commands)

        aggregate_data_path = pth.join(pth.dirname(__file__), "data")
        aggregated_data.to_csv(pth.join(aggregate_data_path, "Aggregate_Motor_Data_RCBD.csv"), index = False)
    
    if mcmc_ready:

        #Load in the datset
        aggregate_data_path = pth.join(pth.dirname(__file__), "data")
        aggregated_data = pd.read_csv(pth.join(aggregate_data_path, "Aggregate_Motor_Data_RCBD.csv"))

        #Instantiate the model
        h_model = HierarchicalModel(aggregated_data)
        #Fit the model
        h_model.fit(sd_multiplier=3)

        #If the model ran without divergences, save result
        if h_model.trace.sample_stats['diverging'].sum() == 0:
            h_model.trace.to_netcdf("data/Accepted_trace.nc")

        #Summarize fitted model
        print(az.summary(h_model.trace, var_names=["R", "K", "b", "c", "J"], round_to = 4))
        print(az.summary(h_model.trace, var_names = ["R_m", "K_m", "b_m", "c_m", "J_m"]))

        #Trace plot
        az.plot_trace(h_model.trace, var_names = ["R", "K", "b", "c", "J"], legend=True)
        plt.show()

        #Plot pairs of group and motor parameters
        az.plot_pair(h_model.trace, var_names = ["R", "R_m"], kind="kde", divergences = True)
        az.plot_pair(h_model.trace, var_names = ["K", "K_m"], kind="kde", divergences = True)
        az.plot_pair(h_model.trace, var_names = ["b", "b_m"], kind="kde", divergences = True)
        az.plot_pair(h_model.trace, var_names = ["c", "c_m"], kind="kde", divergences = True)
        az.plot_pair(h_model.trace, var_names = ["J", "J_m"], kind="kde", divergences = True)
        plt.show()

    if compute_nominal_params:

        #Prepare to write the parameters to a file
        param_file = "data/parameters.txt"
        f = open(param_file, 'w')

        #Set the number of simulations and initialize
        num_sims = 100
        motors, blocks, runs, params_global, params_motor_list, MultiDataReshape = InitializeSimulations("data/Aggregate_Motor_Data_RCBD.csv",
                                                                                                         "data/Accepted_trace.nc", num_sims= num_sims)

        #Run through the motors and get simulations for each block at command 255
        command = 255
            
        #Instantiate global-level model
        global_model = MotorModel(params_global)

        global_idx_stack = []
        
        for id, motor in enumerate(motors):
            
            #Instantiate global-level model
            motor_model = MotorModel(params_motor_list[id])

            motor_ind_idx_stack = []

            models = [global_model, motor_model]

            print("Simulating motor {}".format(motor))
            f.write("Motor {}\n".format(motor))

            run_command = lambda global_out_stack, motor_out_stack, models, data, sr_values, run:RunCommandForParams(global_out_stack, motor_out_stack, models, data, sr_values, run, command)
            RunBlocks(global_idx_stack, motor_ind_idx_stack, models, motor, blocks, runs, MultiDataReshape, multi_speed_commands, run_command)

            best_motor_params = PackOutputList(motor_ind_idx_stack, for_params = True, dict_to_pattern = params_motor_list[id])
            WriteParamsToFile(best_motor_params, f)
            print("...done.")

        print("Simulations finished, no more motors.")
        global_params_dict = PackOutputList(global_idx_stack, for_params=True, dict_to_pattern = params_global)
        f.write("Global Model\n")
        WriteParamsToFile(global_params_dict, f)

        f.close()

    if tune_filter:

        #Load my global and motor-level parameters
        params_global, params_motor_list = OpenParamsFromFile("data/parameters.txt", ["R", "K", "b", "c", "J"])

        #Load in my test data
        data_folder = pth.join(pth.dirname(__file__), "data/Disturbance_Tests")
        test_data = [MotorData(cell, filepath = data_folder) for cell in os.listdir(data_folder)]

        #Define the motor labels, and number of runs
        motors = ['A', 'B', 'C', 'D']
        runs = list(range(1,6))

        #Reshape my motor test data into a dict
        test_data_reshape = {motor:[] for motor in motors}
        for data in test_data:
            filename_segmented = data.filename.split('_')
            motor_label = filename_segmented[3][-1]
            if motor_label in motors:
                test_data_reshape[motor_label].append(data)


        #Create the filepath for the observer data plot figures to go
        #Filepath(s) for saving plots
        plot_path = pth.join(pth.dirname(__file__), "filter plots")
        speed_plot_path = pth.join(plot_path, "speed")

        #make the path if it doesn't exist
        if not pth.exists(plot_path):
            #Specifically make the path to position vs speed data
            os.mkdir(plot_path)
            os.mkdir(speed_plot_path)

        #Instantiate the filter
        alpha = 0.01 #filter gain, higher numbers indicate more trust in new measurements
        filter = EWMA(alpha)

        for run in runs:

            for command in multi_speed_commands:

                #Create subplots with labels, appropriate size (10x7 inches)
                fig_speed, ax_speed = plt.subplots(2,2)
                fig_speed.set_size_inches(10,7) 

                #Set overall title
                fig_speed.suptitle("Speeds for Run = {}, Command = {} \n alpha = {}".format(run, command, alpha))

                for id, motor in enumerate(motors):

                    row_idx = id // 2
                    col_idx = id - (row_idx * 2)

                    sr_values = test_data_reshape[motor][0].get_step_windows(multi_speed_commands)

                    #Extract observed_states
                    times = sr_values[run][command]['Time'].values
                    positions = sr_values[run][command]['Position'].values
                    speed_init_compute = np.diff(positions) / np.diff(times)
                    speeds = np.concatenate([[0], speed_init_compute])

                    #Set title
                    ax_speed[row_idx, col_idx].set_title(motor)

                    #Enable grid lines
                    ax_speed[row_idx, col_idx].grid(True)

                    #Set x and y labels on outside edge
                    if col_idx == 0:
                        ax_speed[row_idx, col_idx].set_ylabel('Speed (rad/s)')
                    if row_idx == 1:
                        ax_speed[row_idx, col_idx].set_xlabel('Time (s)')

                    #Plot observed values
                    ax_speed[row_idx, col_idx].plot(times, speeds, label = "Observed")

                    #Make predictions
                    predictions = filter.predict_batch(speeds)

                    #Plot predicted values
                    ax_speed[row_idx, col_idx].plot(times, predictions, label = "Filter")
            
            #Create legend, removing duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig_speed.legend(by_label.values(), by_label.keys())

            #Save figure, plot results. Start by replacing decimals in gains with underscores
            speed_filename = "Speed_Plot_Run_{}_PWM_{}_alpha{}.jpg".format(run, command, alpha)
            fig_speed.savefig(pth.join(speed_plot_path, speed_filename))

            plt.show()


if __name__ == "__main__":

    main()