import numpy as np
from qNav.stepsize import StepSize
from qNav.policy import ImplicitPolicy

class Agent:
    
    
    def __init__(self, 
                 env, 
                 qfunction,
                 memory,
                 gamma = 0.95, 
                 epsilon = StepSize(mode='linear', init_value=0.8, decay=1e-3), 
                 alpha = StepSize(mode = 'constant', init_value=0.5), 
                 seed = 121314,
                 verbose = False):
        self.env = env 
        self.Q = qfunction
        self.gamma = gamma
        self.memory = memory
        self.alpha = alpha
        self.epsilon = epsilon
        self.stat_memory = np.zeros(self.memory)
        self.curr_len = None
        self.num_episodes = 0
        self.verbose = verbose
        self.pi = ImplicitPolicy(self.Q, seed=seed) 
        
    @property
    def noise_thr(self):
        return self.env.n_thr
        
    def store_Q(self, file):
        np.save(file, self.Q.data, allow_pickle=False)

    def load_Q(self, file):
        self.Q.data = np.load(file, allow_pickle=False)
        
    def update(self, state, action, delta):
        self.Q.update(state, action, self.alpha() * delta)
        
    def set_init_position(self, init_x, init_y):
        # Only for simulations
        self.env.init_position[0] = init_x
        self.env.init_position[1] = init_y
        
        
    def update_lr(self):
        self.num_episodes += 1
        self.alpha.next()
        self.epsilon.next()
        
    def step(self, horizon = None, callback = None, clip_position = False, test=False): 
        steps, total_reward, path, n_zeros, terminated = self._step(horizon = horizon, 
                                                                    callback = callback, 
                                                                    clip_position = clip_position,
                                                                    test = test)
        self.update_lr()        
        return steps, total_reward, path, n_zeros, terminated

    def reset(self):
        self.stat_memory = np.zeros(self.memory, dtype=np.float64)
