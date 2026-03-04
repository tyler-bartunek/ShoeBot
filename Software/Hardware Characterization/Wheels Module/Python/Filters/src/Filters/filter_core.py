
#Array manipulation
import numpy as np

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
