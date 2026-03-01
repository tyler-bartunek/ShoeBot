#File manipulation
import os
import os.path as pth

#Import MCMC tools
import pymc as pm
import statsmodels.api as sm #For obtaining priors via OLS
import arviz as az #For loading old traces

#Progress bar
from tqdm import tqdm

#import data frame manipulation tools
import pandas as pd

#Array manipulation, random sampling for MC generation of J, expm
import numpy as np

#Import custom classes/tools for simulation purposes
from PostProcessing import MotorData, ReshapeMotorData, GetTimesAndVoltages
from Filters import Remap_P_3D
from StateModels import MotorModel

#Hierarchical bayesian model will live in here
class HierarchicalModel:

    def __init__(self, data:pd.DataFrame):

        self.data = data

    def fit(self, sd_multiplier = 10):

        #Otain my priors and hyperpriors
        priors = self.get_priors()
        R, R_se = priors["R"]
        K, K_se = priors["K"]
        b, b_se = priors["b"]
        c, c_se = priors["c"]
        J, J_se = priors["J"]

        #Observation noise in voltage measurements
        #V_sigma_obs = 4 * self.data["Voltage"].std()
        V_sigma_obs = 0.05 #STD of voltage at each command was 0.003

        #Build the categorical level identifiers
        motor_idx = pd.Categorical(self.data['Motor']).codes
        n_motors = len(self.data['Motor'].unique())

        #Initialize my actual internal model object for parameter fitting
        with pm.Model() as model:
            #Hyperpriors
            #R
            R_global = pm.Normal("R", mu = R, sigma = sd_multiplier * R_se)
            sigma_R = pm.HalfNormal("sigma_R", sigma = sd_multiplier * R_se)
            #K
            K_global = pm.Normal("K", mu = K, sigma = sd_multiplier * K_se)
            sigma_K = pm.HalfNormal("sigma_K", sigma = sd_multiplier * K_se)
            #b
            b_global = pm.Normal("b", mu = b, sigma = sd_multiplier * b_se)
            sigma_b = pm.HalfNormal("sigma_b", sigma = sd_multiplier * b_se)
            #c
            c_global = pm.Normal("c", mu = c, sigma = sd_multiplier * c_se)
            sigma_c = pm.HalfNormal("sigma_c", sigma = sd_multiplier * c_se)
            #J
            J_global = pm.Normal("J", mu = J, sigma = sd_multiplier * J_se)
            sigma_J = pm.HalfNormal("sigma_J", sigma = sd_multiplier * J_se)
            #V, response
            V_sigma = pm.HalfNormal("sigma", sigma = V_sigma_obs)

            #Offsets
            R_offset = pm.HalfNormal("R_off", sigma = 1, shape = n_motors)
            K_offset = pm.HalfNormal("K_off", sigma = 1, shape = n_motors)
            b_offset = pm.HalfNormal("b_off", sigma = 1, shape = n_motors)
            c_offset = pm.HalfNormal("c_off", sigma = 1, shape = n_motors)
            J_offset = pm.HalfNormal("J_off", sigma = 1, shape = n_motors)

            #Priors
            R_m = pm.Deterministic("R_m", R_global + R_offset * sigma_R)
            K_m = pm.Deterministic("K_m", K_global + K_offset * sigma_K)
            b_m = pm.Deterministic("b_m", b_global + b_offset * sigma_b)
            c_m = pm.Deterministic("c_m", c_global + c_offset * sigma_c)
            J_m = pm.Deterministic("J_m", J_global + J_offset * sigma_J)


            #primitives -> composites
            constant_term = c_m[motor_idx] + J_m[motor_idx] * R_m[motor_idx]
            tau_term = b_m[motor_idx] * R_m[motor_idx] + K_m[motor_idx] ** 2
            i_term = K_m[motor_idx] + R_m[motor_idx]
            omega_term = K_m[motor_idx] - b_m[motor_idx]

            #Declare current, speed, and time_constant
            current = self.data['Current'].values
            speed = self.data['Speed'].values
            time_constant = self.data['Time Constant'].values

            #compute the deterministic
            V_pred = i_term * current + omega_term * speed + tau_term * time_constant - constant_term

            #Actual observed data
            actual_voltage = self.data['Voltage'].values
            V_obs = pm.Normal("V_obs", mu = V_pred, sigma = V_sigma, observed = actual_voltage)

            #Run the NUTS sampler
            trace = pm.sample(target_accept = 0.95)

        self.model = model
        self.trace = trace
    
    
    def get_priors(self) -> dict:

        #Compute R and K, begin by dividing i_ss and V by \omega_xx
        RK_fit = self.fit_R_K(self.data['Current'].values, self.data['Speed'].values, 
                                  self.data['Voltage'].values)
        R = RK_fit.params[1]
        K = RK_fit.params[0]

        #Compute b and c from nominal value of K and Current
        bc_fit = self.fit_b_c(K, self.data['Current'].values, self.data['Speed'].values)
        b = bc_fit.params[1]
        c = bc_fit.params[0]

        #Extract se estimates
        R_se = RK_fit.bse[1]
        K_se = RK_fit.bse[0]
        b_se = bc_fit.bse[1]
        c_se = bc_fit.bse[0]

        #Obtain J based on computed priors for b, R, and K. 
        J, J_se = self.compute_J({"R":(R,R_se), "K": (K,K_se), "b": (b,b_se)}, self.data["Time Constant"].values)

        priors = {"R":(R,R_se), "K":(K, K_se), "b":(b, b_se), "c":(c, c_se), "J":(J, J_se)}

        return priors
    
    ################# OLS and MC fitting for getting priors #################
    def fit_b_c(self, K_nominal, current_data, speed_data):

        friction = K_nominal * current_data
        speed_data = sm.add_constant(speed_data)

        return sm.OLS(friction, speed_data).fit()
    
    def fit_R_K(self, current_data, speed_data, volt_data):

        i_per_speed = current_data / speed_data
        i_per_speed = sm.add_constant(i_per_speed)

        v_per_speed = volt_data / speed_data

        return sm.OLS(v_per_speed, i_per_speed).fit()
    
    def compute_J(self, param_se:dict, tau, N = 10000) -> tuple:

        #Extract expected value and SE for each parameter
        R, R_se = param_se["R"]
        b, b_se = param_se["b"]
        K, K_se = param_se["K"]

        #Draw N times from the assumed gaussian distribution each parameter belongs to
        R_draws = np.random.normal(R, R_se, N)
        b_draws = np.random.normal(b, b_se, N)
        K_draws = np.random.normal(K, K_se, N)

        #Compute N average J values, averaged over all values of tau
        tau = np.asarray(tau)
        J_draws = np.mean(((b_draws * R_draws + K_draws ** 2) * tau[:, None]) / R_draws, axis=0)

        # Visually confirmed that this ends up looking approximately gaussian
        # plt.hist(J_draws, bins = 50)
        # plt.show()

        return np.mean(J_draws), np.std(J_draws)    


#############################################################################################################################
######################################### Computation/Simulator Functions ###################################################
#############################################################################################################################
def InitializeSimulations(compiled_data_filename:str, trace_filename:str, data_folder:str, num_sims:int = 100):

    #Load in the steady-state response datset
    aggregated_data = pd.read_csv(compiled_data_filename)

    #Extract motor names
    motors = aggregated_data['Motor'].unique()
    blocks = aggregated_data['Block'].unique()
    runs = aggregated_data['Run'].unique()

    #Load in randomly drawn params for models
    params_global, params_motor_list = DrawParams(trace_filename, motors, num_sims = num_sims)

    #Re-load my motor data, whole thing because I want to manipulate my step response data
    multi_speed_files = os.listdir(data_folder)
    MultiData = [MotorData(cell, filepath=data_folder) for cell in multi_speed_files]

    #Reshape MultiData
    MultiDataReshape = ReshapeMotorData(MultiData, motors, blocks)

    return motors, blocks, runs, params_global, params_motor_list, MultiDataReshape

def RunBlocks(output_list_global:list, output_list_motors:list, models:list[MotorModel], motor:str, blocks:list, runs:list, 
              motor_data_dict:dict[str,dict[int,MotorData]], commands:list, CalculateRuns):

    for block in tqdm(blocks):

        sr_values = motor_data_dict[motor][block].get_step_windows(commands)

        for run in tqdm(runs):
            #Still needs some work around the edges to get this to fit nice, but I think we're off to a good start. May need
            #to define some lambdas within the respective if statements in main.
            CalculateRuns(output_list_global, output_list_motors, models, motor_data_dict[motor][block], sr_values, run)

def RunCommandForParams(global_idx_list:list, motor_ind_idx_list:list, models:list[MotorModel], data:MotorData, 
                        sr_values:dict[int, dict[int, pd.DataFrame]], run:int, command:int = 255) -> None:

    for_params = True
    global_model = models[0]
    motor_model = models[1]

    times, voltages = GetTimesAndVoltages(data, sr_values)

    resid_global = SimulateForResiduals(global_model, sr_values, run, command, times, voltages, for_params)
    resid_motor = SimulateForResiduals(motor_model, sr_values, run, command, times, voltages, for_params)

    dist_global = ComputeDistance(resid_global)
    dist_motor = ComputeDistance(resid_motor)

    idx_best_params_global = np.argmin(dist_global)
    idx_best_params_motor = np.argmin(dist_motor)

    global_idx_list.append(idx_best_params_global)
    motor_ind_idx_list.append(idx_best_params_motor)


def SimulateForResiduals(model:MotorModel, sr_values:dict[int,dict[int,pd.DataFrame]], run:int, command:int, times:np.ndarray,
                   voltages:np.ndarray, for_params:bool = True, num_sims:int = 100):
    
    #I feel ok hard-coding initial_states here because I know this is well-behaved for now
    initial_states = np.zeros((num_sims, 2, 1))

    positions = sr_values[run][command]['Position'].values
    observed = positions.copy()

    if not for_params:
        speed_init_compute = np.diff(positions) / np.diff(times)
        speeds = np.concatenate([initial_states[0,1,:], speed_init_compute])

        observed = np.block([[positions.T], [speeds.T]])

    predicted = model.predict_batch(initial_states, times, voltages, num_draws= num_sims)

    residuals = ComputeResiduals(observed, predicted)

    return residuals


#############################################################################################################################
############################################# Helper/Packing Functions ######################################################
############################################################################################################################# 
def DrawParams(trace_filename:str, motors:list, num_sims = 100) -> tuple[dict,list[dict]]:

    #Trace of the posterior currently in use, model with no divergences
    trace = az.from_netcdf(trace_filename)
    #This stacks all the chains together
    posterior_samples = trace.posterior.stack(sample=("chain", "draw"))

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

    params_global = {"R":R_draws, 
                         "K":K_draws,
                         "b":b_draws, 
                         "c":c_draws,
                         "J":J_draws}
    
    params_motors_list = []
    
    for id, motor in enumerate(motors):

        params_motor = {"R":R_m_draws[id, :], 
                        "K":K_m_draws[id, :],
                        "b":b_m_draws[id, :], 
                        "c":c_m_draws[id, :],
                        "J":J_m_draws[id, :]}
        
        params_motors_list.append(params_motor)

    return params_global, params_motors_list

def FetchIterationLists(data:pd.DataFrame) -> tuple:

    return data["Motor"].unique(), data["Block"].unique(), data["Run"].unique()

def MeanBestParams(indices:list[int], params_dict:dict[str,float]) -> dict[str,float]:

    #Don't remove duplicates from list, want params that work more often to be weighed greater in final result
    #Initialize output to zero
    ideal_params = {param_copy:0 for param_copy in params_dict.keys()}

    for param in params_dict.keys():
        ideal_params[param] = np.mean([params_dict[param][i] for i in indices])

    return ideal_params

def ComputeResiduals(observed:np.ndarray, predicted:np.ndarray, state:int = 0) -> np.ndarray:

    if len(observed.shape) > 1:
        return observed[None, :, :] - predicted
    if len(observed.shape) == 1:
        return observed[None, :] - predicted[:, state, :]
    else:
        raise ValueError("Unexpected dimensions for observed values")


def ComputeBestQ(Q_est:np.ndarray) -> np.ndarray:

    Q_hat = np.median(Q_est, axis = 0)

    return ForcePSD(Q_hat)

def ForcePSD(symmetric_matrix:np.ndarray, floor = 1e-16) -> np.ndarray:

    vals, vecs = np.linalg.eigh(symmetric_matrix)
    vals = np.clip(vals, floor, None)

    return vecs @ np.diag(vals) @ vecs.T

def ComputeDistance(residuals:np.ndarray):

    return np.sqrt(np.mean(residuals ** 2, axis = 1))

def PackOutputList(output_list:list, for_params:bool = True, dict_to_pattern = None):

    if for_params:
        return MeanBestParams(output_list, dict_to_pattern)
    
    else:
        stack = np.stack(output_list)
        return ComputeBestQ(stack)
