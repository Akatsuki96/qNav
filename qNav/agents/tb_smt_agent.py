import numpy as np

from qNav.agents.smt_agent import SMTAgent

class TBAgent(SMTAgent):
    def build_features(self, raw_obs, s_thr, action = None, prev_features = None):
        whiff = raw_obs >= s_thr
        intermittency = np.sum(whiff) / len(raw_obs)
        avg_intensity_during_whiff = raw_obs[whiff].mean() if np.any(whiff) else 0
        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
            'improvement' : 0.0,
            't_blank' : 0
        }
        if prev_features is not None:
            features['improvement'] = features['avg_int'] - prev_features['avg_int']
            features['t_blank'] = prev_features['t_blank'] + 1 if raw_obs[-1] < s_thr else 0
        return features
                
