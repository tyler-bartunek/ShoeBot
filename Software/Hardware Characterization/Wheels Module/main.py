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

    # data_filename = "motor_data_romeo_Motor{}_Block_{}_{}".format(motor, block_number, mmddyyyy)
    data_filename = "motor_data_romeo_Motor{}_{}".format(motor, mmddyyyy)

    #data_folder = pth.join(pth.dirname(__file__), "data/RCBD_Trials")
    data_folder = pth.join(pth.dirname(__file__), "data/Observer_Disturbance_Tests")

    #Flags that control what the program is doing: collecting data, providing early visualizations, or preprocessing data
    collecting_data = False
    plot_raw_data = False
    preprocess_data = False

    #Model identification
    mcmc_ready = False
    compute_nominal_params = False

    #Observer 
    tune_observer = False
    test_observer = True

    #Post-plotting
    create_plots = False

    #lists of commands, to be used in plotting and processing steps
    multi_speed_commands = [31, 63, 95, 127, 159, 191, 223, 255]

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

    if tune_observer:

        #Load in my parameters
        params_global, params_motor_list = OpenParamsFromFile('data/parameters.txt', ["R", "K", "b", "c", "J"])

        #Filepath(s) for saving plots
        observer_plot_path = pth.join(pth.dirname(__file__), "observer_plots")
        pos_plot_path = pth.join(observer_plot_path, "position")
        speed_plot_path = pth.join(observer_plot_path, "speed")

        #make the path if it doesn't exist
        if not pth.exists(observer_plot_path):
            #Specifically make the path to position vs speed data
            os.mkdir(observer_plot_path)
            os.mkdir(pos_plot_path)
            os.mkdir(speed_plot_path)

        #Load in my data to test on
        multi_speed_files = os.listdir(data_folder)
        MultiData = [MotorData(cell, filepath=data_folder) for cell in multi_speed_files]

        #Which command to test
        command = 255

        # Get my steady-state file just to get labels
        #Load in the steady-state response datset
        aggregate_data_path = pth.join(pth.dirname(__file__), "data")
        aggregated_data = pd.read_csv(pth.join(aggregate_data_path, "Aggregate_Motor_Data_RCBD.csv"))

        #Extract motor names
        motors = aggregated_data['Motor'].unique()
        blocks = aggregated_data['Block'].unique() 
        runs = aggregated_data['Run'].unique()

        #Reshape MultiData, split into the training and validation data
        MultiDataReshape = ReshapeMotorData(MultiData, motors, blocks)

        #Initialize my gains to test: just for fun, shouldn't really cost much
        #Initially 1 and 0.01 had semi-decent results
        L1 = 1.0
        L2 = 0.01
        gains = np.array([[L1, 0], [0, L2]])

        #We'll just check with the global model on each motor, cause screw it
        global_model = MotorModel(params_global)
        global_observer = StateObserver(global_model, gains)

        initial_states = np.zeros((2, 1))

        for block in blocks:

            for run in runs:

                fig_pos, ax_pos = plt.subplots(2,2)
                fig_speed, ax_speed = plt.subplots(2,2)

                fig_pos.set_size_inches(10,7) #Set to be 10x7 inches
                fig_speed.set_size_inches(10,7) 

                fig_pos.suptitle("Position for Block = {}, Run = {} \n L1 = {}, L2 = {}".format(block, run, L1, L2))
                fig_speed.suptitle("Speeds for Block = {}, Run = {} \n L1 = {}, L2 = {}".format(block, run, L1, L2))

                for id, motor in enumerate(motors):

                    row_idx = id // 2
                    col_idx = id - (row_idx * 2)

                    #Screw it, we ball: observer with the same gains but different params
                    motor_model = MotorModel(params_motor_list[id])
                    motor_observer = StateObserver(motor_model, gains)

                    ax_pos[row_idx, col_idx].set_title('Motor {}'.format(motor))
                    ax_speed[row_idx, col_idx].set_title('Motor {}'.format(motor))

                    data_obj = MultiDataReshape[motor][block]
                    sr_values = data_obj.get_step_windows(multi_speed_commands)

                    times, voltages = GetTimesAndVoltages(data_obj, sr_values[run][command])

                    #Extract observed_states
                    positions = sr_values[run][command]['Position'].values
                    speed_init_compute = np.diff(positions) / np.diff(times)
                    speeds = np.concatenate([initial_states[1], speed_init_compute])

                    #Enable gridlines
                    ax_pos[row_idx, col_idx].grid(True)
                    ax_speed[row_idx, col_idx].grid(True)

                    #Plot positions and speeds over time
                    ax_pos[row_idx, col_idx].plot(times, positions, label = "Observed")
                    ax_speed[row_idx, col_idx].plot(times, speeds, label = "Observed")

                    observed = np.block([[positions.T], [speeds.T]])

                    predictions_global = global_observer.predict_batch(voltages, observed, times)
                    predictions_motor = motor_observer.predict_batch(voltages, observed, times)

                    #Plot predictions
                    ax_pos[row_idx, col_idx].plot(times,predictions_global[0,:], label = "Global Model")
                    ax_speed[row_idx, col_idx].plot(times,predictions_global[1,:], label = "Global Model")
                    
                    ax_pos[row_idx, col_idx].plot(times,predictions_motor[0,:], label = "Motor Model")
                    ax_speed[row_idx, col_idx].plot(times,predictions_motor[1,:], label = "Motor Model")

                    #Only set axis labels on outside edge
                    if (col_idx == 0):  
                        ax_pos[row_idx, col_idx].set_ylabel('Position (rad)')  
                        ax_speed[row_idx, col_idx].set_ylabel('Speed (rad/s)')
                    if (row_idx == 1):
                        ax_pos[row_idx, col_idx].set_xlabel('Time (s)')
                        ax_speed[row_idx, col_idx].set_xlabel('Time (s)')

                #Create legend, removing duplicate labels
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                fig_pos.legend(by_label.values(), by_label.keys())
                fig_speed.legend(by_label.values(), by_label.keys())

                #Save figure, plot results. Start by replacing decimals in gains with underscores
                L1_file_sub_start, L2_file_sub_start = "{}".format(L1), "{}".format(L2)
                L1_file_sub, L2_file_sub = L1_file_sub_start.replace(".","_"), L2_file_sub_start.replace(".","_")
                position_filename = "Position_Plot_Block_{}_Run_{}_L1_{}_L2_{}.jpg".format(block, run, L1_file_sub, L2_file_sub)
                speed_filename = "Speed_Plot_Block_{}_Run_{}_L1_{}_L2_{}.jpg".format(block, run, L1_file_sub, L2_file_sub)
                fig_pos.savefig(pth.join(pos_plot_path,position_filename))
                fig_speed.savefig(pth.join(speed_plot_path, speed_filename))
            plt.show()

    if test_observer:

        #Load my global and motor-level parameters
        params_global, params_motor_list = OpenParamsFromFile("data/parameters.txt", ["R", "K", "b", "c", "J"])

        #Load in my test data
        data_folder = pth.join(pth.dirname(__file__), "data/Observer_Disturbance_Tests")
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

        #Instantiate filter gains
        L1 = 1.0
        L2 = 0.01
        test_gains = np.array([[L1, 0.0], [0.0, L2]])

        #Instantiate the initial states
        initial_states = np.zeros((2,1))

        #Create the filepath for the observer data plot figures to go
        #Filepath(s) for saving plots
        observer_plot_path = pth.join(pth.dirname(__file__), "observer_plots/test_results")
        pos_plot_path = pth.join(observer_plot_path, "position")
        speed_plot_path = pth.join(observer_plot_path, "speed")

        #make the path if it doesn't exist
        if not pth.exists(observer_plot_path):
            #Specifically make the path to position vs speed data
            os.mkdir(observer_plot_path)
            os.mkdir(pos_plot_path)
            os.mkdir(speed_plot_path)

        for run in runs:

            for command in multi_speed_commands:

                #Create subplots with labels, appropriate size
                fig_pos, ax_pos = plt.subplots(2,2)
                fig_speed, ax_speed = plt.subplots(2,2)

                fig_pos.set_size_inches(10,7) #Set to be 10x7 inches
                fig_speed.set_size_inches(10,7) 

                fig_pos.suptitle("Position for Run = {}, Command = {} \n L1 = {}, L2 = {}".format(run, command, L1, L2))
                fig_speed.suptitle("Speeds for Run = {}, Command = {} \n L1 = {}, L2 = {}".format(run, command, L1, L2))

                for id, motor in enumerate(motors):

                    row_idx = id // 2
                    col_idx = id - (row_idx * 2)

                    motor_model = MotorModel(params_motor_list[id])
                    motor_observer = StateObserver(motor_model, test_gains)

                    sr_values = test_data_reshape[motor][0].get_step_windows(multi_speed_commands)

                    #Extract inputs
                    times = sr_values[run][command]['Time']
                    voltages = sr_values[run][command]['PWM'] * 12.2 / 255

                    #Extract observed_states
                    positions = sr_values[run][command]['Position'].values
                    speed_init_compute = np.diff(positions) / np.diff(times)
                    speeds = np.concatenate([initial_states[1], speed_init_compute])

                    observed = np.block([[positions.T], [speeds.T]])

                    #Set titles
                    ax_pos[row_idx, col_idx].set_title(motor)
                    ax_speed[row_idx, col_idx].set_title(motor)

                    #Enable grid lines
                    ax_pos[row_idx, col_idx].grid(True)
                    ax_speed[row_idx, col_idx].grid(True)

                    #Set x and y labels on outside edge
                    if col_idx == 0:
                        ax_pos[row_idx, col_idx].set_ylabel('Position (rad)')
                        ax_speed[row_idx, col_idx].set_ylabel('Speed (rad/s)')
                    if row_idx == 1:
                        ax_pos[row_idx, col_idx].set_xlabel('Time (s)')
                        ax_speed[row_idx, col_idx].set_xlabel('Time (s)')

                    #Plot observed values
                    ax_pos[row_idx, col_idx].plot(times, positions, label = "Observed")
                    ax_speed[row_idx, col_idx].plot(times, speeds, label = "Observed")

                    #Make predictions
                    predictions = motor_observer.predict_batch(voltages, observed, times)

                    #Plot predicted values
                    ax_pos[row_idx, col_idx].plot(times, predictions[0,:], label = "Motor Observer")
                    ax_speed[row_idx, col_idx].plot(times, predictions[1,:], label = "Motor Observer")
            
            #Create legend, removing duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig_pos.legend(by_label.values(), by_label.keys())
            fig_speed.legend(by_label.values(), by_label.keys())

            #Save figure, plot results. Start by replacing decimals in gains with underscores
            L1_file_sub_start, L2_file_sub_start = "{}".format(L1), "{}".format(L2)
            L1_file_sub, L2_file_sub = L1_file_sub_start.replace(".","_"), L2_file_sub_start.replace(".","_")
            position_filename = "Position_Plot_Run_{}_PWM_{}_L1_{}_L2_{}.jpg".format(run, command, L1_file_sub, L2_file_sub)
            speed_filename = "Speed_Plot_Run_{}_PWM_{}_L1_{}_L2_{}.jpg".format(run, command, L1_file_sub, L2_file_sub)
            fig_pos.savefig(pth.join(pos_plot_path,position_filename))
            fig_speed.savefig(pth.join(speed_plot_path, speed_filename))

            plt.show()


    
    if create_plots:

        #Trace of the posterior currently in use, model with no divergences
        trace = az.from_netcdf("data/Accepted_trace.nc")

        #Trace plot
        az.plot_trace(trace, var_names = ["R", "K", "b", "c", "J"], legend=True)
        plt.show()
        #Plot pairs of group and motor parameters
        az.plot_pair(trace, var_names = ["R", "R_m"], kind="kde", divergences = True)
        az.plot_pair(trace, var_names = ["K", "K_m"], kind="kde", divergences = True)
        az.plot_pair(trace, var_names = ["b", "b_m"], kind="kde", divergences = True)
        az.plot_pair(trace, var_names = ["c", "c_m"], kind="kde", divergences = True)
        az.plot_pair(trace, var_names = ["J", "J_m"], kind="kde", divergences = True)
        plt.show()

        #This stacks all the chains together for monte carlo simulation
        posterior_samples = trace.posterior.stack(sample=("chain", "draw"))

        #Draw global parameter values
        num_sims = 100
        idx = np.random.choice(posterior_samples.sample.size, num_sims, replace=False)
        R_draws = posterior_samples["R"].values.flatten()[idx]
        K_draws = posterior_samples["K"].values.flatten()[idx]
        b_draws = posterior_samples["b"].values.flatten()[idx]
        c_draws = posterior_samples["c"].values.flatten()[idx]
        J_draws = posterior_samples["J"].values.flatten()[idx]

        #Draw motor_level parameter values
        R_m_draws = posterior_samples["R_m"].values[:, idx]
        K_m_draws = posterior_samples["K_m"].values[:, idx]
        b_m_draws = posterior_samples["b_m"].values[:, idx]
        c_m_draws = posterior_samples["c_m"].values[:, idx]
        J_m_draws = posterior_samples["J_m"].values[:, idx]

        #Pick a subset of data files to run simulations with. 
        #Since I don't need to compute Q using R anymore, I should just be able to pick one from each motor.

        raise NotImplementedError


if __name__ == "__main__":

    main()