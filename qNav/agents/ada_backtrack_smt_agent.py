import numpy as np
from math import ceil

from qNav.agents.q_agent import QAgent
from qNav.utils import count_consecutive

class AdaBCKAgent(QAgent):
    def __init__(self, min_mem = 5, init_mem = 10, seed=12, rnd_on_zero = True, **kwargs):
        super().__init__(**kwargs)
        self.first_obs = True
        self.obs_smt = False
        self.non_zero_observed = False
        self.rnd_on_zero = rnd_on_zero
        self.init_mem = init_mem
        self.rnd_state = np.random.RandomState(seed)
        self.backtrack_stack = []
        self.backtracking = False
        self.INV_ACTIONS = [1, 0, 3, 2]
        self.iact = None
        self.in_void = False
        self.min_mem = min_mem
        self.tb_len = 0
        self.curr_len = init_mem #self.memory
        self.last_blank = 0
                
    def sensitivity_thr(self, raw_obs):        
        return np.max([0.5 * self.stat_memory[-self.curr_len:].mean(), self.noise_thr])

    def _set_clen(self):
        self.curr_len = self.last_blank + 2 if self.last_blank > self.min_mem else self.min_mem

                
    def _update_sensing_memory(self, raw_obs):
        if self.first_obs:
            self.stat_memory = raw_obs.copy().reshape(-1)
            self.first_obs = False
        else:
            self.stat_memory = np.delete(self.stat_memory, 0)
            self.stat_memory = np.concatenate((self.stat_memory, raw_obs.copy())).reshape(-1)
        if self.stat_memory.mean() > 1e-20:
            self.non_zero_observed = True


    def get_state(self, raw_obs, action, prev_features):
        self._update_sensing_memory(raw_obs)            
        s_thr = self.sensitivity_thr(raw_obs)
        if prev_features is not None:
            if self.obs_smt and raw_obs[-1] < s_thr:
                self.tb_len += 1
            elif self.obs_smt and self.tb_len > 0 and raw_obs[-1] >= s_thr:
                self.obs_smt = False
                self.last_blank = self.tb_len
                self._set_clen()
                self.tb_len = 0  
            elif not self.obs_smt and raw_obs[-1] >= s_thr: 
                self.obs_smt = True     
        else:
            # blks = count_consecutive(self.stat_memory, s_thr)
            # blk_len = ceil(np.mean(blks)) if len(blks) > 0 else 0 
            self.curr_len = self.init_mem #blk_len if blk_len > self.min_mem else self.min_mem #blk_len if blk_len > self.min_mem else self.min_mem

        features = self.process_observation(self.stat_memory[-self.curr_len:], s_thr, action = action, prev_features = prev_features)
        self.fmap.update(raw_obs, features, s_thr)
        f_idx = self.fmap.to_index(list(features.values()))
        return f_idx, features, s_thr

    def _backtrack_action(self, state, action, features, test):
        if features['avg_int'] >0.0:
            self.in_void = False
            self.backtrack_stack = []
            self.backtracking = False
            self.iact=None
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif len(self.backtrack_stack) == 0:
            if test and self.rnd_on_zero:
                return self.rnd_state.randint(0, 4) 
            return self.pi.sample_epsilon_greedy(state, self.epsilon())  

#            action = self.rnd_state.randint(0, 4) 
#            self.iact = self.INV_ACTIONS[action] if self.iact is None else None
#            return action
            
        return self.backtrack_stack.pop()         

        
    def backtrack_policy(self, raw_obs, state, features, test, action, s_thr):
        if action is None:
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        if self.backtracking:
            return self._backtrack_action(state, action, features, test)# TBD
        last_is_blank = self.stat_memory[-1] < s_thr
        if features['avg_int'] > 0.0 and not last_is_blank:
            self.in_void = False
            self.non_zero_observed = True
            self.backtrack_stack = []
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif last_is_blank and features['avg_int'] > 0.0:
            self.backtrack_stack.append(self.INV_ACTIONS[action])
            return self.pi.sample_epsilon_greedy(state, self.epsilon())            
        elif features['avg_int'] == 0.0:
            self.in_void = True
            if not self.non_zero_observed:
                
               # return self.rnd_state.randint(0, 4) #
                return self.pi.sample_epsilon_greedy(state, self.epsilon())#self.rnd_state.randint(0, 4)
            self.backtracking = True
            if len(self.backtrack_stack) == 0:
                return self.INV_ACTIONS[action]
            return self.backtrack_stack.pop()
        raise NotImplementedError("[--] Unhandled case!\n\t[--] features: {}\n\tmemory: {}\traw_obs: {}".format(features, self.stat_memory, raw_obs))

    def get_action(self, raw_obs, state, features, test, action, s_thr):
        return self.backtrack_policy(raw_obs, state, features, test, action, s_thr)

    def reset(self):
        self.stat_memory = np.zeros(self.memory, dtype=np.float64)
        self.first_obs = True
        self.backtrack_stack = []
        self.backtracking = False
        self.non_zero_observed = False
        self.iact = None
        self.tb_len = 0
        self.curr_len = self.init_mem# self.memory
        self.last_blank = 0
        self.obs_smt = False
        self.in_void = False
