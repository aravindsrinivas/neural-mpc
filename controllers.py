import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        pass

    def get_action(self, state):
        return env.action_space.sample()
	#pass


class MPCcontroller(Controller):
    def __init__(self, env, dyn_model, horizon=5, cost_fn=None, num_simulated_paths=10):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        trajs = [env.action_space.sample() for _ in range(self.horizon*self.num_simulated_paths)].reshape((self.num_simulated_paths, self.horizon, -1))
        #TODO: action_dim #n_paths*horizon*action_dim
        accum_states = np.zeros((self.num_simulated_paths, self.horizon, state.shape[0]))
        accum_next_states = np.zeros((self.num_simulated_paths, self.horizon, state.shape[0]))
        states = np.ones((num_simulated_paths,1))*state # n_paths*state_dim
        for time_idx in range(self.horizon):
            actions = trajs[:, time_idx, :] # n_paths * action_dim
            next_states = self.dyn_model.predict(states, actions)
            accum_states[:, time_idx, :] = states
            accum_next_states[:, time_idx, :] = next_states
            next_states = states.copy()
            cost = trajectory_cost_fn(self.cost_fn, accum_next_states, trajs, accum_states)
            action = trajs[np.argmax(cost), 0, :]
            return action
