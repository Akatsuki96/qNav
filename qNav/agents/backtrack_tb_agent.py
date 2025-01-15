import numpy as np

from qNav.agents.q_agent import QAgent

class BCKTBAgent(QAgent):
    def __init__(self, seed=12, rnd_on_zero = True, **kwargs):
        super().__init__(**kwargs)
        self.first_obs = True
        self.non_zero_observed = False
        self.rnd_on_zero = rnd_on_zero
        self.rnd_state = np.random.RandomState(seed)
        self.backtrack_stack = []
        self.backtracking = False
        self.INV_ACTIONS = [1, 0, 3, 2]
        self.iact = None
                
    def sensitivity_thr(self, raw_obs):        
        return np.max([0.5 * self.stat_memory.mean(), self.noise_thr])
                
                
    def _update_sensing_memory(self, raw_obs):
        if self.first_obs:
            self.stat_memory = raw_obs.copy().reshape(-1)
            self.first_obs = False
        else:
            self.stat_memory = np.delete(self.stat_memory, 0)
            self.stat_memory = np.concatenate((self.stat_memory, raw_obs.copy())).reshape(-1)
        if self.stat_memory.mean() > 1e-20:
            self.non_zero_observed = True

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
    
    def get_state(self, raw_obs, action, prev_features):
        self._update_sensing_memory(raw_obs)            
        s_thr = self.sensitivity_thr(raw_obs)
        features = self.process_observation(self.stat_memory, s_thr, action = action, prev_features = prev_features)
        self.fmap.update(raw_obs, features, s_thr)
        f_idx = self.fmap.to_index(list(features.values()))
        return f_idx, features, s_thr

    def _backtrack_action(self, state, action, features):
        if features['avg_int'] >0.0:
            self.backtrack_stack = []
            self.backtracking = False
            self.iact=None
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif len(self.backtrack_stack) == 0:
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
#            action = self.rnd_state.randint(0, 4) if self.iact is None else self.iact
#            self.iact = self.INV_ACTIONS[action] if self.iact is None else None
#            return action
            
        return self.backtrack_stack.pop()         

        
    def backtrack_policy(self, raw_obs, state, features, test, action, s_thr):
        if action is None:
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        if self.backtracking:
            return self._backtrack_action(state, action, features)# TBD
        last_is_blank = self.stat_memory[-1] < s_thr
        if features['avg_int'] > 0.0 and not last_is_blank:
            self.non_zero_observed = True
            self.backtrack_stack = []
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif last_is_blank and features['avg_int'] > 0.0:
            self.backtrack_stack.append(self.INV_ACTIONS[action])
            return self.pi.sample_epsilon_greedy(state, self.epsilon())            
        elif features['avg_int'] == 0.0:
            if not self.non_zero_observed:
                return self.rnd_state.randint(0, 4)
            self.backtracking = True
            if len(self.backtrack_stack) == 0:
                return self.INV_ACTIONS[action]
            return self.backtrack_stack.pop()
        raise NotImplementedError("[--] Unhandled case!\n\t[--] features: {}\n\tmemory: {}\traw_obs: {}".format(features, self.stat_memory, raw_obs))

    def get_action(self, raw_obs, state, features, test, action, s_thr):
        if self.rnd_on_zero and len(self.stat_memory[self.stat_memory >= s_thr]) == 0:
            return self.rnd_state.randint(0, 4)
        return self.backtrack_policy(raw_obs, state, features, test, action, s_thr)

    def reset(self):
        self.stat_memory = np.zeros(self.memory, dtype=np.float64)
        self.first_obs = True
        self.backtrack_stack = []
        self.backtracking = False
        self.non_zero_observed = False
        self.iact = None
