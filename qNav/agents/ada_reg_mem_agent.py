import numpy as np

from math import ceil

from qNav.agents.smt_agent import SMTAgent
from qNav.utils import count_consecutive
INVERSE_ACTION = 4

from itertools import product


mems = [3, 5, 10, 20, 30, 40, 50]

coord = [0, 1, 2, 3]

actions = product(coord, mems)

def_mem_actions = {}

for (i, act) in enumerate(actions):
    def_mem_actions[i] = act


class AdaRegMemAgent(SMTAgent):
    def __init__(self, min_mem = 5, init_mem = 10, mem_actions = def_mem_actions, seed=12, rnd_on_zero= True, **kwargs):
        super().__init__(seed, rnd_on_zero, **kwargs)
        self.first_obs = True
        self.tb_len = 0
        self.init_mem = init_mem
        self.last_blank = 0
        self.mem_actions = mem_actions
        self.min_mem = min_mem
        self.curr_len = init_mem #self.memory
        self.rnd_state = np.random.RandomState(seed)
        self.mem_used = []

    def build_features(self, raw_obs, s_thr, action = None, prev_features = None):
        whiff = raw_obs >= s_thr
        intermittency = np.sum(whiff) / len(raw_obs)
        avg_intensity_during_whiff = raw_obs[whiff].mean() if np.any(whiff) else 0
        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
        }
        features['zeros'] = 0
        if prev_features is None and np.all(raw_obs < s_thr):
            features['zeros'] = 1
        elif prev_features is not None and np.all(raw_obs < s_thr):
            features['zeros'] = prev_features['zeros'] + 1
        return features

    def _init_used_memory(self, o_thr):
        # blks = count_consecutive(self.stat_memory, o_thr)
        # blk_len = ceil(np.mean(blks)) if len(blks) > 0 else 0 
        # self.curr_len = blk_len if blk_len > self.min_mem else self.min_mem #blk_len if blk_len > self.min_mem else self.min_mem
        self.curr_len = self.init_mem

    def _set_clen(self):
        self.curr_len = self.last_blank  if self.last_blank > self.min_mem else self.min_mem

    def process_observation(self, raw_obs, s_thr, action = None, prev_features = None):
        if prev_features is None:
#            self._init_used_memory(s_thr)
            return self.build_features(self.stat_memory[-self.init_mem:], s_thr, action = action, prev_features = prev_features)

#        else:
#            if raw_obs[-1] < s_thr:
#                self.tb_len += 1
#                if self.greedy:
#                self._set_clen()
#            else:
#                if self.tb_len > 0 and self.tb_len > self.curr_len:
#                    self.last_blank = self.tb_len
#                elif self.tb_len > 0:
#                    self.last_blank -= 1
#                self._set_clen()
#                self.tb_len = 0         
        return self.build_features(self.stat_memory[-action[1]:], s_thr, action = action, prev_features = prev_features)
                
    def get_action(self, raw_obs, state, features, test, action, s_thr):
 #       mem = self.stat_memory[-self.curr_len:]
#        if self.rnd_on_zero and test and len(mem[mem >= s_thr]) == 0:
#        if test and len(mem[mem >= s_thr]) == 0:
#            return self.rnd_state.randint(0, 4)
        return self.pi.sample_epsilon_greedy(state, self.epsilon())
          
                
    def _step(self, horizon = None, callback = None, clip_position=False, test = False):
        path = []
        raw_obs, terminated = self.env.reset(self.memory)
        self.fmap.reset()
        if terminated:
            return 0, np.inf, path, 0, terminated
        n_zeros = 0
        state, features, o_thr = self.get_state(raw_obs, None, None)
        if features['avg_int'] == 0.0:
            n_zeros += 1
        k, tot_reward = 0, 0.
        if callback is not None:
            callback(self, state, raw_obs, None, features, o_thr)
        action = prev_action = None
        while horizon is None or k < horizon:
            #print(self.env.current_position)
            action = self.get_action(raw_obs, state, features, test, prev_action, o_thr) 
            t_action = self.mem_actions[action]
            path.append(t_action)
            raw_obs, reward, terminated = self.env.step(t_action[0], clip_position = clip_position)
            next_state, features, o_thr = self.get_state(raw_obs, t_action, features)
            if features['avg_int'] == 0.0:
                n_zeros += 1
            next_action = self.next_action(next_state)
            if not test:
                target = reward + self.gamma * self.Q.of(next_state, next_action)
                delta = target - self.Q.of(state, action)
                self.update(state, action, delta)
            if callback is not None:
                callback(self, next_state, raw_obs, action, features, o_thr)
            k += 1
            tot_reward += reward
            state = next_state
            prev_action = action
            if terminated:
                break
            
        return k, tot_reward, path, n_zeros, terminated
           

    def reset(self):
        super().reset()
        self.tb_len = 0
        self.curr_len = self.init_mem #self.memory
        self.last_blank = 0