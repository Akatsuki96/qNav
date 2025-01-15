import numpy as np

from qNav.utils import default_reward

COORDS = {
    0 : np.asarray([ 1, 0], dtype=np.int32), # right
    1 : np.asarray([-1, 0], dtype=np.int32), # left
    2 : np.asarray([0,  1], dtype=np.int32), # down
    3 : np.asarray([0, -1], dtype=np.int32), # up
}

INVERSE_ACTION = 4
SAME_ACTION = 5
LO_ACTION = 6 # Left orthogonal
RO_ACTION = 7 # Right othogonal

INV_ACTIONS = [1, 0, 3, 2]

class Environment:
    
    def __init__(self, 
                 data, 
                 init_position, 
                 terminal_position,
                 reward_function = default_reward,
                 velocity = 10,
                 eps = (20, 20),
                 n_thr = 3e-6,
                 bounds : np.ndarray = None,
                 seed : int = 1212
                 ):
        self.data = data
        self.rnd_state = np.random.RandomState(seed)
        self.bounds = bounds # [[min_x, max_x], [min_y, max_y]]
        self.velocity = velocity
        self.n_thr = n_thr
        if bounds is None:
            self.bounds = np.asarray([
                [0, self.data.shape[0]],
                [0, self.data.shape[1]],
            ])
        self.init_position = init_position
        self.current_position = init_position.copy()
        self.terminal_position = terminal_position
        self.reward_function = reward_function
        self.prev_action = None
        self.eps = eps
        self.lb = 0
        self.ub = 0
        self.t = 0
        self.MAX_T = self.data.shape[2] 
                
    def _is_in_bounds(self, pos):
        return self.bounds[0, 0] <= pos[0] < self.bounds[0, 1] and self.bounds[1, 0] <= pos[1] < self.bounds[1, 1]

    def _is_in_discretization(self, pos):
        return 0 <= pos[0] < self.data.shape[0] and 0 <= pos[1] < self.data.shape[1]

    def _is_near_enough(self, pos):
        return (abs(pos[0] - self.terminal_position[0]) <= self.eps[0]) and abs(pos[1] - self.terminal_position[1]) <= self.eps[1]
        
    def clip_position(self, pos):
        new_pos = pos.copy()
        if pos[0] < self.bounds[0, 0]:
            delta_x = self.bounds[0, 0] - pos[0]
            new_pos[0] = self.bounds[0, 1] - delta_x
        elif pos[0] >= self.bounds[0, 1]:
            delta_x = pos[0] - self.bounds[0, 1]
            new_pos[0] = self.bounds[0, 0] + delta_x
        if pos[1] < self.bounds[1, 0]:
            delta_x = self.bounds[1, 0] - pos[1]
            new_pos[1] = self.bounds[1, 1] - delta_x
        elif pos[1] >= self.bounds[1, 1]:
            delta_x = pos[1] - self.bounds[1, 1]
            new_pos[1] = self.bounds[1, 0] + delta_x
        return new_pos
            
    def get_action(self, action_idx : int):
        if (action_idx == INVERSE_ACTION or action_idx == SAME_ACTION) and self.prev_action is None:
            idx = self.rnd_state.choice(4)
            return COORDS[idx]
        elif action_idx == INVERSE_ACTION:
            return COORDS[INV_ACTIONS[self.prev_action]]
        elif action_idx == SAME_ACTION:
            return COORDS[self.prev_action]
        return COORDS[action_idx]

        
        
    def play_action(self, action_idx : int, clip_position : bool = True):
        v = self.get_action(action_idx = action_idx)
        self.current_position = self.current_position + self.velocity * v
        if clip_position:
            self.current_position = self.clip_position(self.current_position)
        return self._get_value()

    def step(self, action, clip_position : bool = False): 
        raw_obs = self.play_action(action, clip_position)
        if action != INVERSE_ACTION and action !=SAME_ACTION:
            self.prev_action = action

        terminated = self._is_near_enough(self.current_position)
        reward = self.reward_function(self.t, raw_obs, terminated, self) #1.0 if terminated else 0.0 
        return raw_obs, reward, terminated

    def _get_value(self): 
        '''get num_obs values in current position'''
        pos = self.current_position
        if self._is_in_discretization(self.current_position):
            data = self.data[pos[0], pos[1], self.lb : self.lb + 1]
        else:
            data = np.full(1, 0.0, dtype=np.float64) # if agent is out of bounds, return zeros
        self.lb += 1
        if self.lb >= self.MAX_T:
            self.lb = 0        
        return data
        
        
    def reset(self, num_obs : int):
        self.current_position = self.init_position.copy()
        self.lb = num_obs
        self.prev_action = None
        if not self._is_in_discretization(self.current_position):
            return np.zeros(num_obs),False
        data = self.data[self.current_position[0], self.current_position[1], :num_obs]
        return data, self._is_near_enough(self.current_position)
        
        