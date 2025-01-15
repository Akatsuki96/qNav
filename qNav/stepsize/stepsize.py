import numpy as np


class StepSize:
    
    def __init__(self, mode : str, init_value : float, end_value : float = 0.0, decay : float = 1e-2) -> None:
        assert mode in ['constant', 'linear', 'exponential', 'sqrt']
        self.update = getattr(StepSize, "_" + mode)
        self.init_value = init_value
        self.alpha = init_value
        self.mode = mode
        self.end_value = end_value
        self.decay = decay
        self.num_episode = 0

    def _constant(self) -> float:
        return self.init_value
    
    def _linear(self) -> float:
        return max(self.init_value / (1.0 + (self.decay * self.num_episode)), self.end_value)
    
    def _exponential(self) -> float:
        return max(self.init_value * np.exp(-self.decay * self.num_episode), self.end_value)
    
    def _sqrt(self) -> float:
        return max(self.init_value /np.sqrt(self.num_episode), self.end_value)
        
    def __call__(self) -> float:
        return self.alpha

    def next(self) -> None:
        self.num_episode += 1
        self.alpha = self.update(self)   

    def __repr__(self) -> str:
        return "init: {}\tcurrent: {}\tmode: {}\tdecay: {}\tepisodes: {}".format(self.init_value, self.alpha, self.mode, self.decay, self.num_episode)