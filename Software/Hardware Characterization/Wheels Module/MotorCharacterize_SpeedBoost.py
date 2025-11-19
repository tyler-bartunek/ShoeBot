"""
MotorCharacterize_SpeedBoost
"""

#Standard library imports: basic functionality
import os
import os.path as pth
import pandas as pd
import numpy as np

#For line fitting
import statsmodels.api as sm

#Serial communication and post-processing
from SerialStreamParser import GatherAndStore
from PostProcessing import MotorData, ParseMotorData, ReshapeMotorData, GetTimesAndVoltages

#Progress bar
from tqdm import tqdm

#generating unique filename for date and time
from datetime import datetime

#MCMC code and simulations
from MCMC_Simulator import HierarchicalModel, InitializeSimulations, RunBlocks, RunCommandForParams, RunCommandsForQ, PackOutputList
from StateModels import MotorModel

#Kalman Filter, MLE fitting
from Kalman import Filter, StateObserver, EWMA, MLE

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

    data_folder = pth.join(pth.dirname(__file__), "data/RCBD_Trials")
    # data_folder = pth.join(pth.dirname(__file__), "data/Observer_Disturbance_Tests")

    #Flags that control what the program is doing: collecting data, providing early visualizations, or preprocessing data
    collecting_data = False
    plot_raw_data = False
    preprocess_data = False

    #Model identification
    mcmc_ready = False
    compute_nominal_params = False
    compute_process_noise = False
    tune_kalman = False

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

    if compute_process_noise:

        #Initialize my global and motor_level process noise covariance matrices
        Q_global = np.zeros((2,2))
        Q_motor = np.zeros((2,2,4)) #One for each motor
        Q_global_stack = []
        Q_motor_stack = []

        #Set the number of simulations
        num_sims = 100

        motors, blocks, runs, params_global, params_motor_list, MultiDataReshape = InitializeSimulations("data/Aggregate_Motor_Data_RCBD.csv",
                                                                                                         "data/Accepted_trace.nc", num_sims= num_sims)
            
        #Instantiate global-level model
        global_model = MotorModel(params_global)
        
        for id, motor in enumerate(motors):
            
            #Instantiate motor-level model
            motor_model = MotorModel(params_motor_list[id])
            
            models = [global_model, motor_model]

            print("Simulating Motor {}".format(motor))

            Q_individual_motor_list = []

            #Iterate over all experimental blocks
            run_commands = lambda output_list_global, output_list_motors, models, data, sr_values, run:RunCommandsForQ(output_list_global, output_list_motors, models, data, sr_values, run, multi_speed_commands)
            RunBlocks(Q_global_stack, Q_individual_motor_list, models, motor, blocks, runs, MultiDataReshape[motor][block], multi_speed_commands, run_commands) 

            #Convert to stack, get median, force PSD, append
            Q_ind_motor = PackOutputList(Q_individual_motor_list, for_params = False)
            Q_motor_stack.append(Q_ind_motor)

        
        #Take median of these values, guarantee PSD, stack final motor 3D array
        Q_global = PackOutputList(Q_global_stack)
        Q_motor = np.stack(Q_motor_stack)

        #Save arrays
        np.save('data/Q_motor.npy', Q_motor)
        np.save('data/Q_global.npy', Q_global)

    if tune_observer:

        #Load in my parameters
        params_global, params_motor_list = OpenParamsFromFile('data/parameters.txt', ["R", "K", "b", "c", "J"])

        #Filepath(s) for saving plots
        observer_plot_path = pth.join(pth.dirname(__file__), "observer_plots/with_ewma")
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
        L2 = 0.1
        gains = np.array([[L1, 0], [0, L2]])

        #Instantiate global observer
        global_model = MotorModel(params_global)
        global_observer = StateObserver(global_model, gains)

        initial_states = np.zeros((2, 1))

        #Attempt exponentially weighted average filtering with observer
        #Higher numbers = higher trust in measurements
        digital_filter = EWMA(0.3)

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

                    #Make predictions
                    predictions_global = global_observer.predict_batch(voltages, observed, times, digital_filt = digital_filter)
                    predictions_motor = motor_observer.predict_batch(voltages, observed, times, digital_filt = digital_filter)

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


    if tune_kalman:

        #Load in my starting Q matrices
        Q_0_global = np.load('data/Q_global.npy')
        Q_0_motor_stack = np.load('data/Q_motor.npy')

        #Poor man's logging rev 2, cause we aren't saving the parameter values for the fits the hard way 
        log_file = open('data/R_parameters.txt', 'w')

        #Overall strategy: Load in data, first three blocks for each motor are training, last block is validation. 
        #Use all data for each block, don't break into step responses.

        # Get my steady-state file just to get labels
        #Load in the steady-state response datset
        aggregate_data_path = pth.join(pth.dirname(__file__), "data")
        aggregated_data = pd.read_csv(pth.join(aggregate_data_path, "Aggregate_Motor_Data_RCBD.csv"))

        #R looks like it could be proportional to the speed and/or voltage commmand... now to eliminate outliers in omega and theta

        #identify outliers
        Q1_omega = aggregated_data['Var_Omega'].quantile(0.25)
        Q3_omega = aggregated_data['Var_Omega'].quantile(0.75) 
        IQR_omega = Q3_omega - Q1_omega
        lower_bound = Q1_omega - 1.5 * IQR_omega
        upper_bound = Q3_omega + 1.5 * IQR_omega

        #Remove outliers from the dataset, along with their columns
        inclusion_criteria = (aggregated_data['Var_Omega'] >= lower_bound) & (aggregated_data['Var_Omega'] <= upper_bound)
        cleaned_var_data = aggregated_data[inclusion_criteria]

        #Fit lines/models to data at global level for R values
        speed = cleaned_var_data['Speed'].values
        speed = sm.add_constant(speed)
        var_omega_line_global = sm.OLS(cleaned_var_data["Var_Omega"].values, speed).fit()
        quadratic_speed = np.concatenate([speed, (cleaned_var_data['Speed'].values ** 2).reshape(len(speed),1)], axis = 1)
        var_theta_line_global = sm.OLS(cleaned_var_data['Var_Theta'].values, quadratic_speed).fit()

        #Pack into dict. list of dicts for motor-level
        R_params_global = {'theta':list(var_theta_line_global.params), 'omega':list(var_omega_line_global.params)}
        R_params_motor_list = []

        #Extract motor names
        motors = aggregated_data['Motor'].unique()
        blocks = aggregated_data['Block'].unique() 
        runs = aggregated_data['Run'].unique()

        multi_speed_files = os.listdir(data_folder)
        MultiData = [MotorData(cell, filepath=data_folder) for cell in multi_speed_files]

        #Reshape MultiData, split into the training and validation data
        MultiDataReshape = ReshapeMotorData(MultiData, motors, blocks)
        train, validate = [], []
        for motor in motors:

            #Fit models
            speed_motor = cleaned_var_data['Speed'][cleaned_var_data['Motor'] == motor].values
            speed_motor = sm.add_constant(speed_motor)
            speed_motor_quadratic = np.concatenate([speed_motor, (cleaned_var_data['Speed'][cleaned_var_data['Motor'] == motor].values ** 2).reshape(len(speed_motor),1)], axis = 1)
            var_omega_line_motor = sm.OLS(cleaned_var_data['Var_Omega'][cleaned_var_data['Motor'] == motor].values, speed_motor).fit()
            var_theta_line_motor = sm.OLS(cleaned_var_data['Var_Theta'][cleaned_var_data['Motor'] == motor].values, speed_motor_quadratic).fit()
            R_params_motor_list.append({'theta':list(var_theta_line_motor.params), 'omega':list(var_omega_line_motor.params)})
                    

            for block in blocks:
                if block <= 3:
                    train.append(MultiDataReshape[motor][block])
                else:
                    validate.append(MultiDataReshape[motor][block])

        TrainDataReshape = ReshapeMotorData(train, motors, list(range(1,4)))
        ValidateDataReshape = ReshapeMotorData(validate, motors, [4])

        #Load my global and motor-level parameters
        params_global, params_motor_list = OpenParamsFromFile("data/parameters.txt", ["R", "K", "b", "c", "J"])

        #Instantiate my global model and the KF for that model
        global_model = MotorModel(params_global)
        global_model_filter = Filter(global_model)

        initial_states = np.zeros((2, 1))

        #Train the model
        for id, motor in enumerate(motors):

            #Instantiate motor model and KF for that model
            motor_model = MotorModel(params_motor_list[id])
            motor_model_filter = Filter(motor_model)

            #Extract starting Q, R_params
            Q_0_motor = Q_0_motor_stack[id]
            R_params_motor = R_params_motor_list[id]

            #Extract blocks for training: not to be confused with the total number of blocks
            blocks = TrainDataReshape[motor].keys()
            log_file.write("Motor {}\n".format(motor))

            for block in tqdm(blocks):

                #Extract training data, including times and voltages during run
                train_data_obj = TrainDataReshape[motor][block]

                sr_values = train_data_obj.get_step_windows(multi_speed_commands)

                for run in tqdm(runs):
                    for command in tqdm(multi_speed_commands):

                        times, voltages = GetTimesAndVoltages(train_data_obj, sr_values[run][command])

                        #Extract observed_states
                        positions = sr_values[run][command]['Position'].values
                        speed_init_compute = np.diff(positions) / np.diff(times)
                        speeds = np.concatenate([initial_states[1], speed_init_compute])

                        observed = np.block([[positions.T], [speeds.T]])

                        # Run simulation to obtain optimal Q and R (parameters in that second case)
                        optimal_Q_global = MLE(global_model_filter, times, voltages, observed, Q_0_global, R_params_global)
                        optimal_Q_motor = MLE(motor_model_filter, times, voltages, observed, Q_0_motor, R_params_motor)

                        #I do need to consider if I want to pass Q and R for the global model back in as new estimates though. I think I do?
                        Q_0_global = optimal_Q_global
                        Q_0_motor = optimal_Q_motor

            np.save('data/Q_optimal_motor_{}.npy'.format(motor), optimal_Q_motor)
            for param in R_params_motor.keys():
                log_file.write("{}: ".format(param))
                for val in R_params_motor[param]:
                    log_file.write("{}, ".format(val))
            log_file.write("\n")

        np.save('data/Q_optimal_global.npy', optimal_Q_global)
        log_file.write("Global Model\n")
        for param in R_params_global.keys():
            log_file.write("{}: ".format(param))
            for val in R_params_global[param]:
                log_file.write("{}, ".format(val))
            log_file.write("\n")

        log_file.close()

    if test_observer:

        #Load my global and motor-level parameters
        params_global, params_motor_list = OpenParamsFromFile("data/parameters.txt", ["R", "K", "b", "c", "J"])

        #Load in my test data
        data_folder = pth.join(pth.dirname(__file__), "data/Observer_Disturbance_Tests")
        test_data = [MotorData(cell, filepath = data_folder) for cell in os.listdir(data_folder)]

        #Define the motor labels, and number of runs: hard-coded cause I'm lazy
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
        L2 = 0.7
        test_gains = np.array([[L1, 0.0], [0.0, L2]])

        #Instantiate the initial states
        initial_states = np.zeros((2,1))

        #Create the filepath for the observer data plot figures to go
        #Filepath(s) for saving plots
        observer_plot_path = pth.join(pth.dirname(__file__), "observer_plots/test_results/with_digital_filt")
        pos_plot_path = pth.join(observer_plot_path, "position")
        speed_plot_path = pth.join(observer_plot_path, "speed")

        #make the path if it doesn't exist
        if not pth.exists(observer_plot_path):
            #Specifically make the path to position vs speed data
            os.mkdir(observer_plot_path)
            os.mkdir(pos_plot_path)
            os.mkdir(speed_plot_path)

        #Introduce pre-filtering
        digital_filter = EWMA(0.01)

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
                    predictions = motor_observer.predict_batch(voltages, observed, times, digital_filt = digital_filter)

                    #Plot EWMA predicted values
                    ax_speed[row_idx, col_idx].plot(times, digital_filter.predict_batch(speeds), label = "Moving Average")

                    #Plot observer predicted values
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