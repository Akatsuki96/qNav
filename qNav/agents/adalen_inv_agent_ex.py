import numpy as np

from qNav.agents.smt_agent import SMTAgent

from math import ceil

from qNav.utils import count_consecutive

INVERSE_ACTION = 4

def build_features(o_thr, raw_observation, prev_feature, action):
    whiff = raw_observation >= o_thr
    avg_odor_during_whiff = raw_observation[whiff].mean() if np.any(whiff) else 0.0
    intermittency = np.sum(whiff) / len(raw_observation)
    labels = ['right', 'left', 'down', 'up']
    features = {
        'intermittency' : intermittency,
        'avg_int' : avg_odor_during_whiff,
        't_blank' : 0,
        'inverse' : 0
    }
    if action is not None:
        features['inverse'] = int(action == INVERSE_ACTION)
    for i in range(len(labels)):
        features[labels[i]] = 0.0
    if action is not None:
        features[labels[action]] = features['avg_int'] - prev_feature['avg_int']
    if prev_feature is not None:
        features['t_blank'] = prev_feature['t_blank'] + 1 if raw_observation[-1] < o_thr else 0
            
    return features


def process_observation(o_thr, raw_observation, agent=None, action=None, prev_features=None):
    if prev_features is not None:
        blanks = count_consecutive(agent.stat_memory, o_thr)
        
        tb = ceil(np.mean(blanks)) if len(blanks) > 0 else 0#prev_features['t_blank'] if prev_features['t_blank'] > 0 else 1
        if tb == 0:
            tb = 1
        return build_features(o_thr, agent.stat_memory[-tb:], prev_features, action)
        
    return build_features(o_thr, agent.stat_memory, prev_features, action)
    

def s_thr(raw, agent):
    return np.max([1/2 * agent.stat_memory.mean(), agent.env.n_thr])

class AdaLenInvAgentEx(SMTAgent):
    def __init__(self, seed=12, rnd_on_zero= True, **kwargs):
        super().__init__(seed, rnd_on_zero, **kwargs)
        self.first_obs = True
        self.rnd_state = np.random.RandomState(seed)
        self._process_observation = process_observation
        self.sensitivity_thr = s_thr
                

