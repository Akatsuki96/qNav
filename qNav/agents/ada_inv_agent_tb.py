import numpy as np

from math import ceil

from qNav.agents.smt_agent import SMTAgent
from qNav.utils import count_consecutive
INVERSE_ACTION = 4

class AdaInvTBAgent(SMTAgent):
    def __init__(self, min_mem = 5, seed=12, rnd_on_zero= True, **kwargs):
        super().__init__(seed, rnd_on_zero, **kwargs)
        self.first_obs = True
        self.tb_len = 0
        self.min_mem = min_mem
        self.last_blank = 0
        self.curr_len = self.memory
        self.rnd_state = np.random.RandomState(seed)
        self.mem_used = []

    def build_features(self, raw_obs, s_thr, action = None, prev_features = None):
        whiff = raw_obs >= s_thr
        intermittency = np.sum(whiff) / len(raw_obs)
        avg_intensity_during_whiff = raw_obs[whiff].mean() if np.any(whiff) else 0
        blank_len = count_consecutive(raw_obs, s_thr)
        avg_blank_len = np.mean(blank_len) if len(blank_len) > 0 else 0.0

        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
            'avg_tb' : self.curr_len#avg_blank_len
        }

        return features

    def _init_used_memory(self, o_thr):
#        blks = count_consecutive(self.stat_memory, o_thr)
#        blk_len = ceil(np.mean(blks)) if len(blks) > 0 else 0 
        self.curr_len = self.memory#blk_len if blk_len > self.min_mem else self.min_mem

    def _set_clen(self):
        self.curr_len = self.last_blank if self.last_blank > self.min_mem else self.min_mem

    def process_observation(self, raw_obs, s_thr, action = None, prev_features = None):
        if prev_features is None:
            self._init_used_memory(s_thr)
        else:
            if raw_obs[-1] < s_thr:
                self.tb_len += 1
            else:
                if self.tb_len > 0:
                    self.last_blank = self.tb_len
                self._set_clen()
                self.tb_len = 0         
        return self.build_features(self.stat_memory[-self.curr_len:], s_thr, action = action, prev_features = prev_features)
                
    def get_action(self, raw_obs, state, features, test, action, s_thr):
        mem = self.stat_memory[-self.curr_len:]
        if self.rnd_on_zero and test and len(mem[mem >= s_thr]) == 0:
            return self.rnd_state.randint(0, 4)
        return self.pi.sample_epsilon_greedy(state, self.epsilon())
                

    def reset(self):
        super().reset()
        self.tb_len = 0
        self.curr_len = self.memory