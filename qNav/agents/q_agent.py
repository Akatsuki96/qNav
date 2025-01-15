import numpy as np

from qNav.stepsize import StepSize
from qNav.agents.agent import Agent

class QAgent(Agent):

    def __init__(self, env, q_function, fmap, **kwargs):
        super().__init__(env, q_function, **kwargs)            
        self.fmap = fmap 
        
    def sensitivity_thr(self, raw_obs):        
        return np.max([0.5 * raw_obs.mean(), self.noise_thr])
    
    def build_features(self, raw_obs, s_thr, action = None, prev_features = None):
        whiff = raw_obs >= s_thr
        intermittency = np.sum(whiff) / len(raw_obs)
        avg_intensity_during_whiff = raw_obs[whiff].mean() if np.any(whiff) else 0
        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
        #    'improvement' : 0.0 
        }
        #if prev_features is not None:
        #    features['improvement'] = features['avg_int'] - prev_features['avg_int']
        return features

    
    def process_observation(self, raw_obs, s_thr, action = None, prev_features = None):
        return self.build_features(raw_obs, s_thr, action = action, prev_features = prev_features)

    def get_state(self, raw_obs, action, prev_features):
        s_thr = self.sensitivity_thr(raw_obs)
        features =self.process_observation(raw_obs, s_thr, action = action, prev_features = prev_features)
        self.fmap.update(raw_obs, features, s_thr)
        fmap_idx = self.fmap.to_index(list(features.values()))
        return fmap_idx, features, s_thr
        
    def next_action(self, next_state):
        return self.pi.sample_greedy(next_state)

            
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
            path.append(action)
            raw_obs, reward, terminated = self.env.step(action, clip_position = clip_position)
            next_state, features, o_thr = self.get_state(raw_obs, action, features)
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
#            if not test and not self.env._is_in_bounds(self.env.current_position):
#                tot_reward -= 100.0
#                tot_reward = horizon * reward
#                break
            prev_action = action
            if terminated:
                break
            
        return k, tot_reward, path, n_zeros, terminated


