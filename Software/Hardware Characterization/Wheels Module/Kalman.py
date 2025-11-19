
#Array manipulation
import numpy as np

#Import tools for log-likelihood optimization
from scipy.optimize import minimize

from StateModels import StateSpaceModel #Note: do not actually use with StateSpaceModel parent class, please
                                        #use a specific child class

class EWMA:
    #Nature of this filter says it only works on one state at a time.

    def __init__(self, alpha:float = 0.0):

        self.alpha = alpha #Default value says to reject the new data outright

    def predict_next(self, current_state:np.ndarray, last_state:np.ndarray):

        return self.alpha * current_state + (1 - self.alpha) * last_state
    
    def predict_batch(self, observations:np.ndarray):

        predictions = np.zeros_like(observations)
        prev_meas = observations[0]

        for idx, observation in enumerate(observations):

            prev_meas = self.predict_next(observation, prev_meas)
            try:
                predictions[idx + 1] = prev_meas
            except IndexError:
                break #all done

        return predictions

class StateObserver:

    def __init__(self, model:StateSpaceModel, gain:np.ndarray):
        self.model = model
        self.model.get_continuous_model()
        self.gain = gain

    def predict_next(self, last_state:np.ndarray, input:np.ndarray, observed:np.ndarray, sample_time = 0.002):
        
        next_state = self.model.predict_next(sample_time, last_state, input)
        innovation = observed.reshape(last_state.shape) - (self.model.C @ next_state.reshape(last_state.shape))

        return next_state + self.gain @ innovation
    
    def predict_batch(self, inputs:np.ndarray, observed_states:np.ndarray, times:np.ndarray, digital_filt:EWMA = None) -> np.ndarray:

        init_state = observed_states[:,0]
        last_state = init_state.copy()
        num_states = last_state.shape[0]

        measured_state = observed_states[:,1]

        dt_vec = np.diff(times)

        predictions = np.zeros((num_states,len(times)))

        for idx, dt in enumerate(dt_vec):

            #Prediction step:Optionally apply digital filter
            measured_state = observed_states[:,idx]
            if digital_filt:      
                measured_state[1] = digital_filt.predict_next(observed_states[1,idx], last_state.reshape((num_states,))[1])
                
            last_state = self.predict_next(last_state.reshape((num_states, 1)), inputs[idx], measured_state, sample_time=dt)
            
            try:
                predictions[:,idx + 1] = last_state.squeeze()
            except IndexError:
                pass #All done
 
        return predictions



class Filter:

    def __init__(self, model:StateSpaceModel):

        self.model = model
        self.model.get_continuous_model()
        self.gain = np.eye(self.model.A_c.shape[1]) #Asssumption that A is 2D and square

    def get_a_priori_state(self, last_state:np.ndarray, input:np.ndarray, sample_time = 0.002):

        return self.model.predict_next(sample_time, last_state, input)
    
    def get_a_priori_P(self, last_P:np.ndarray, Q:np.ndarray, sample_time = 0.002) -> np.ndarray:

        self.model.discretize_model(sample_time, num_draws = 1)
        self.model.A = self.model.A.reshape((self.model.A.shape[1], self.model.A.shape[2]))
        return self.model.A @ last_P @ self.model.A.T + Q
    
    def update_gain(self, P:np.ndarray, R:np.ndarray):

        C = self.model.C 
        self.S = C @ P @ C.T + R
        self.gain = P @ C.T @ np.linalg.inv(self.S)
    
    def update_prediction(self, state_update, observation) -> np.ndarray:

        return state_update + self.gain @ (observation.reshape(state_update.shape) - self.model.C @ state_update)
    
    def get_post_priori_P(self, a_priori_P):

        return (np.eye(self.gain.shape[0]) - self.gain @ self.model.C) @ a_priori_P


#Runs MLE
def MLE(filter:Filter, time_vec:np.ndarray, inputs:np.ndarray, observed_states:np.ndarray, Q0:np.ndarray, R0_params:dict[str,list]) -> tuple[np.ndarray, dict[str,list]]:

    #Assume an error covariance
    P0 = np.array([[100, 0], [0, 1000]])
    
    #Define my cost function
    fitting_cost_function = lambda packed_Q_params: SimulateFilter(filter, time_vec, observed_states[:,0], inputs, observed_states, P0, packed_Q_params, R0_params, return_likelihood= True)

    #Optimize SimulateFilter given a Q, R0_params initial. Find optimal values for each.
    packed_initial_params = PackParams(Q0)
    optimization_result = minimize(fitting_cost_function, packed_initial_params)

    ideal_values = optimization_result.x

    return UnpackParams(ideal_values)

#Runs filter simulation.
def SimulateFilter(filter:Filter, time_vec:np.ndarray, init_state:np.ndarray, inputs:np.ndarray, observed_states:np.ndarray, 
        P0:np.ndarray, packed_Q_params:np.ndarray, R_params:dict[str,list], return_likelihood:bool = False) -> np.ndarray:


    last_state = init_state.copy()
    num_states = last_state.shape[0]

    Q = UnpackParams(packed_Q_params)

    R_theta_coeffs = R_params['theta']
    R_omega_coeffs = R_params['omega']

    dt_vec = np.diff(time_vec)

    prevP = P0.copy()

    predictions = np.zeros((num_states,len(time_vec)))
    innovation = np.zeros_like(predictions)

    S_inv_stack = []
    S_det_stack = []

    for idx, dt in enumerate(dt_vec):

        #Compute R values (needs to be positive at all costs)
        speed = last_state[1]
        R11 = abs(R_theta_coeffs[0] + R_theta_coeffs[1] * speed + R_theta_coeffs[2] * (speed ** 2))
        R22 = abs(R_omega_coeffs[0] + R_omega_coeffs[1] * speed)
        R = np.array([[R11, 0], [0, R22]])

        #Prediction step
        predict = filter.get_a_priori_state(last_state.reshape((num_states, 1)), inputs[idx], dt)
        P = filter.get_a_priori_P(prevP, Q, dt)

        #Correction step: update last state and prediction vector
        filter.update_gain(P, R)
        last_state = filter.update_prediction(predict.reshape((num_states,1)), observed_states[:,idx]).squeeze()
        try:
            predictions[:,idx + 1] = last_state
        except IndexError:
            pass #All done
        prevP = filter.get_post_priori_P(P)

        #Compute innovation, extract innovation covariance, and likelihood at each time step 
        S_inv_stack.append(np.linalg.inv(filter.S))
        S_det_stack.append(np.linalg.det(filter.S))

    innovation = observed_states - predictions
    S_inv = np.stack(S_inv_stack, axis = 0)
    S_det = np.stack(S_det_stack, axis = 0)

    #Likelihood assumes residuals come from a multivariate gaussian
    #exp(-0.5 * [x-mu].T @ covariance @ [x-mu]) / sqrt((2 * pi)^k * det(covariance))
    likelihood_coeff_denom = np.sqrt(((2 * np.pi) ** num_states) * S_det)
    likelihood_exponent = -0.5 * np.einsum('in,nij,jn->n', innovation[:,1:],S_inv,innovation[:,1:])
    likelihood = np.sum(np.log(likelihood_coeff_denom) - likelihood_exponent) #Negative log likelihood


    if return_likelihood:
        return likelihood
    else: 
        return predictions
    


def Remap_P_3D(P:np.ndarray, model:StateSpaceModel, sampling_time = 0.002, num_draws = 100):

    model.discretize_model(sampling_time, num_draws = num_draws)
    return model.A @ P @ np.transpose(model.A, (0,2,1))

def PackParams(Q:np.ndarray) -> np.ndarray:

    #Packs the parameters for my initial guess into a format compatible with scipy optimize inputs
    #Converts Q into its Cholesky factors to ensure PSD
    Q_factors = np.linalg.cholesky(Q)
    parameters = [Q_factors[0,0], Q_factors[1,0], Q_factors[1,1]]

    return np.array(parameters)

def UnpackParams(parameters:np.ndarray) -> tuple[np.ndarray, dict[str,list]]:

    Q_factors = np.zeros((2,2))
    Q_factors[0,0] = parameters[0]
    Q_factors[1,0] = parameters[1]
    Q_factors[1,1] = parameters[2]

    Q = Q_factors @ Q_factors.T.conj()

    return Q