import numpy as np
from scipy.linalg import expm


class StateSpaceModel:

    def __init__(self, params:dict):

        self.params = params

    def get_continuous_model(self, num_draws = 1) -> None:

        self.A_c = np.zeros((2,2, num_draws))
        self.B = np.empty((2,1, num_draws))
        self.C = np.empty((2,2, num_draws))
        self.D = np.empty((2,1, num_draws))

    def discretize_model(self, sampling_time, num_draws = 1) -> None:

        self.A = self.A_c * sampling_time
        self.B = self.B_c * sampling_time

    def predict_next(self, sample_time, current_state:np.ndarray, current_input:np.ndarray, num_draws = 1) -> np.ndarray:
        """
        Returns the next set of outputs for the state-space model given current inputs and sampling
        time. 
        """
        #Get my state transition matrix for given deltaT
        self.get_discrete_model(sample_time, num_draws)

        #Compute the next state based on a current state estimate and the current system input
        next_state = np.einsum('ijk,jlk->ilk', self.A, current_state) + np.einsum('ijk,jlk->ilk', self.B, current_input)

        return np.einsum('ijk,jlk->ilk', self.C, next_state) + np.einsum('ijk,jlk->ilk', self.D, current_input)
    
    def predict_batch(self, initial_conditions:np.ndarray, time_data:np.ndarray, voltage_data:np.ndarray, num_draws = 1) -> np.ndarray:
        """
        Predicts a batch of state estimates given initial conditions and a dataset
        """

        #Get time deltas for each data point
        deltaT = np.diff(time_data)

        #Get continuous model
        self.get_continuous_model(num_draws)

        #Initialize next_state as initial conditions, assuming 
        if len(initial_conditions.shape) > 2:
            num_states = initial_conditions.shape[1]
        else:
            num_states = initial_conditions.shape[0]
        next_state = initial_conditions.reshape(num_draws, num_states, 1)

        #Initialize output
        predictions = np.zeros((num_draws, num_states, len(time_data)))

        for idx, ts in enumerate(deltaT):

            next_state = self.predict_next(ts, next_state, voltage_data[idx], num_draws)
            try:
                predictions[:,:,idx+1] = next_state.squeeze()
            except IndexError: #We've hit the end of our time array
                pass

        return predictions
    
class MotorModel(StateSpaceModel):

    def __init__(self, params:dict):

        self.R = np.asarray(params["R"])
        self.K = np.asarray(params["K"])
        self.b = np.asarray(params["b"])
        self.J = np.asarray(params["J"])
        self.c = np.asarray(params["c"])

    def get_continuous_model(self, num_draws = 1) -> None:

        #Initialize A_c, B_c, C, and D
        self.A_c = np.zeros((num_draws,2,2))
        self.B_c = np.zeros((num_draws,2,1))
        self.g_c = np.zeros((num_draws,2,1))

        self.C = np.eye(2)
        self.D = np.zeros((num_draws,2,1))
        
        #Compute the bottom term of the A matrix
        JR_term = self.J * self.R
        self.A_c[:,0,1] = 1.0
        self.A_c[:,1,1] = -(self.b * self.R + self.K ** 2) / JR_term

        #B
        self.B_c[:,1,0] = self.K / JR_term
        self.g_c[:,1,0] = self.c / self.J


    def discretize_model(self, sampling_time, num_draws = 1) -> None:


        #Convert A and B to discrete domain
        A_cT = self.A_c * sampling_time
        B_cT = self.B_c * sampling_time
        G_cT = self.g_c * sampling_time

        self.A = np.zeros_like(self.A_c)
        self.B = np.zeros_like(self.B_c)

        #Create augmented matrix, obtain Pade approximation
        MT = np.block([[A_cT, B_cT, G_cT],
                       [np.zeros((num_draws,2,4))],])
        MT_d = expm(MT)

        self.A = MT_d[:,0:2,0:2]
        self.B = MT_d[:,0:2,2:3]
        self.g_d = MT_d[:,0:2,3:4]

    def predict_next(self, sample_time, current_state:np.ndarray, current_input:np.ndarray, num_draws = 1) -> np.ndarray:
        """
        Returns the next set of outputs for the state-space model given current inputs and sampling
        time. 
        """
        #Get my state transition matrix for given deltaT
        self.discretize_model(sample_time, num_draws)

        #Compute the next state based on a current state estimate and the current system input
        return self.A @ current_state + self.B * current_input - self.g_d

class EncoderModel(StateSpaceModel):

    """This instance of state model takes the empty dict as params"""

    def get_discrete_model(self, sampling_time) -> None:

        #Model that treats position and speed as states, acceleration as input
        self.A = np.array([[1.0, sampling_time], [0.0, 1.0]])
        self.B = np.array([[0.5 * sampling_time ** 2], [sampling_time]]) 

        #C and D are again trivial
        self.C = np.eye(2)
        self.D = np.zeros_like(self.B) 