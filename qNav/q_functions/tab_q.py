import numpy as np

class TabularQ:
    
    def __init__(self, shape):
        self.data = np.full(shape,0.6, dtype=np.float64)
        self.n_states = np.prod(shape[:-1])
        self.n_actions = shape[-1]    
        
    def of(self, state, action):
        return self.data[state][action]
    
    def all_values(self, state):
   #     print("[--] state: {}".format(state))
        return self.data[state]
    
    def update(self, state, action, delta):
        self.data[state][action] += delta
    
    