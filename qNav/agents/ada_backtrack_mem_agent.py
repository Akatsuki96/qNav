import numpy as np
from math import ceil

from qNav.agents.q_agent import QAgent
from qNav.utils import count_consecutive

from itertools import product

mems = [3, 5, 10, 20, 30, 40, 50]

coord = [0, 1, 2, 3]

actions = product(coord, mems)

def_mem_actions = {}

for (i, act) in enumerate(actions):
    def_mem_actions[i] = act

class AdaBCKMemAgent(QAgent):
    def __init__(self,  init_mem = 10, seed=12, mem_actions = def_mem_actions, rnd_on_zero = True, **kwargs):
        super().__init__(**kwargs)
        self.first_obs = True
        self.obs_smt = False
        self.non_zero_observed = False
        self.in_void = False
        self.rnd_on_zero = rnd_on_zero
        self.init_mem = init_mem
        self.rnd_state = np.random.RandomState(seed)
        self.backtrack_stack = []
        self.backtracking = False
        self.mem_actions =mem_actions
        self.INV_ACTIONS = self.get_inv(mem_actions) #[1, 0, 3, 2]
        self.iact = None
        self.tb_len = 0
        self.curr_len = init_mem #self.memory
        self.last_blank = 0
        
    def _get_action_idx(self, mem_actions, coo, mem):
        for (k, v) in mem_actions.items():
            if v[0] == coo and mem == v[1]:
                return k
        
    def get_inv(self, mem_actions):
        inv_actions = {}
        INV = [1, 0, 3, 2]
        for (k, v) in mem_actions.items():
            inv_actions[v] = self._get_action_idx(mem_actions, INV[v[0]], v[1])
            
        return inv_actions
                

    def get_admissible_actions(self, mem):
        admissible_actions = []
        for (k, v) in self.mem_actions.items():
            if v[1] == mem:
                admissible_actions.append(k)
        return admissible_actions
                
  
    def _backtrack_action(self, state, action, features, test):
        if features['avg_int'] >0.0:
            self.in_void = False
            self.backtrack_stack = []
            self.backtracking = False
            self.iact=None
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif len(self.backtrack_stack) == 0:
#            if test and self.rnd_on_zero:
#            _, mem = self.mem_actions[action]
            adm_actions = self.get_admissible_actions(1)
            return self.rnd_state.choice(adm_actions,1)[0] 
#            return self.pi.sample_epsilon_greedy(state, self.epsilon())  

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
            self.non_zero_observed = True
            self.backtrack_stack = []
            return self.pi.sample_epsilon_greedy(state, self.epsilon())
        elif last_is_blank and features['avg_int'] > 0.0:
            t_action = self.mem_actions[action]
            self.backtrack_stack.append(self.INV_ACTIONS[t_action])
            return self.pi.sample_epsilon_greedy(state, self.epsilon())            
        elif features['avg_int'] == 0.0:
            self.in_void = True
            if not self.non_zero_observed:
                
               # return self.rnd_state.randint(0, 4) #
                return self.pi.sample_epsilon_greedy(state, self.epsilon())#self.rnd_state.randint(0, 4)
            self.backtracking = True
            if len(self.backtrack_stack) == 0:
                t_action = self.mem_actions[action]
                t_action[1] = 1                
                return self.INV_ACTIONS[t_action]
            return self.backtrack_stack.pop()
        raise NotImplementedError("[--] Unhandled case!\n\t[--] features: {}\n\tmemory: {}\traw_obs: {}".format(features, self.stat_memory, raw_obs))

    def get_action(self, raw_obs, state, features, test, action, s_thr):
        return self.backtrack_policy(raw_obs, state, features, test, action, s_thr)

    def process_observation(self, raw_obs, s_thr, action = None, prev_features = None):
        if prev_features is not None:
            return self.build_features(self.stat_memory[-action[1]:], s_thr, action = action, prev_features = prev_features)
         
        return self.build_features(self.stat_memory[-self.init_mem:], s_thr, action = action, prev_features = prev_features)
                    
                
                
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
        self.stat_memory = np.zeros(self.memory, dtype=np.float64)
        self.first_obs = True
        self.backtrack_stack = []
        self.backtracking = False
        self.non_zero_observed = False
        self.iact = None
        self.in_void = False
        self.tb_len = 0
        self.curr_len = self.init_mem# self.memory
        self.last_blank = 0
        self.obs_smt = False