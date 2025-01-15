import numpy as np

from qNav.agents.smt_agent import SMTAgent

INVERSE_ACTION = 4

def build_features(o_thr, raw_observation, prev_feature, action):
    whiff = raw_observation >= o_thr
    avg_odor_during_whiff = raw_observation[whiff].mean() if np.any(whiff) else 0.0
    intermittency = np.sum(whiff) / len(raw_observation)
    features = {
        'intermittency' : intermittency,
        'avg_int' : avg_odor_during_whiff,
        'improvement' : 0.0,
        't_blank' : 0,
        'inverse' : 0
    }
    if action is not None:
        features['inverse'] = int(action == INVERSE_ACTION)
    
    if prev_feature is not None:
        features['improvement'] = features['avg_int'] - prev_feature['avg_int']
        features['t_blank'] = prev_feature['t_blank'] + 1 if raw_observation[-1] < o_thr else 0
            
    return features

class TBInvAgent(SMTAgent):
    def build_features(self, raw_obs, s_thr, action = None, prev_features = None):
        whiff = raw_obs >= s_thr
        intermittency = np.sum(whiff) / len(raw_obs)
        avg_intensity_during_whiff = raw_obs[whiff].mean() if np.any(whiff) else 0
        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
            'improvement' : 0.0 }
        if prev_features is not None:
            features['improvement'] = features['avg_int'] - prev_features['avg_int']
        return features

