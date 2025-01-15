import numpy as np
from math import ceil
from qNav.agents.smt_agent import SMTAgent
from qNav.utils import count_consecutive
INVERSE_ACTION = 4

class AdaLenInvAgent(SMTAgent):
    def __init__(self, seed=12, rnd_on_zero= True, **kwargs):
        super().__init__(seed, rnd_on_zero, **kwargs)
        self.first_obs = True
        self.tb = 0
        self.rnd_state = np.random.RandomState(seed)

    def build_features(self, raw_obs, s_thr, action = None, prev_features = None):
        whiff = raw_obs >= s_thr
        intermittency = np.sum(whiff) / len(raw_obs)
        avg_intensity_during_whiff = raw_obs[whiff].mean() if np.any(whiff) else 0
        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
            'improvement' : 0.0,
            'inverse' : 0
        }
        if action is not None:
            features['inverse'] = int(action == INVERSE_ACTION)
        if prev_feature is not None:
            features['improvement'] = features['avg_int'] - prev_features['avg_int']
        return features



   #def process_observation(o_thr, raw_observation, agent=None, action=None, prev_features=None):
    def process_observation(self, raw_obs, s_thr, action = None, prev_features = None):
        if prev_features is not None:
            blk_lengths = count_consecutive(self.stat_memory, s_thr)
            tb =  ceil(np.mean(blk_lengths)) if len(blk_lengths) > 0 else 1
            if tb == 0:
                tb = 1
            if tb > self.memory:
                return self.build_features(self.stat_memory, s_thr, action = action, prev_features = prev_features)
            return self.build_features(self.stat_memory[-tb:], s_thr, action = action, prev_features = prev_features)
         
        return self.build_features(self.stat_memory, s_thr, action = action, prev_features = prev_features)
                    
                

    def reset(self):
        super().reset()
        self.tb = 0