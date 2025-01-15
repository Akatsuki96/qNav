import numpy as np

class ImplicitPolicy:
    
    def __init__(self, tab_q, seed : int = 1234):
        self.tab_q = tab_q
        self.rnd_state = np.random.RandomState(seed)
        self.n_actions = tab_q.n_actions
        
    def sample_greedy(self, state):
        values = self.tab_q.all_values(state)
        return np.argmax(values)
    
    def sample_epsilon_greedy(self, state, epsilon: float) -> int: 
        if self.rnd_state.rand() <= epsilon:
            return self.rnd_state.choice(self.n_actions)
        return self.sample_greedy(state)
