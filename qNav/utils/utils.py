
import numpy as np

def default_reward(k, raw_obs, terminated, env):
    if terminated:
        return 1.0#100.0
    return -0.001

def count_consecutive(data, thr, filtered=False):
    blank = data < thr if not filtered else data
    blk = []
    c = 0
    obs = False
    init = False
    for x in blank:
        if x and obs:
            c += 1
        elif x and not obs and init:
            obs=True
            c = 1
        elif not x and obs:
            obs = False
            blk.append(c)
            c = 0            
        elif not x and not obs:
            init = True
            
#    if c != 0:
#        blk.append(c)
    return blk    

def count_steps(x1, x2, stepsize=30, rad=30):
    MAXITERS = 100000
    delta = 0
    sng = 1
    while np.abs(x1 - x2) > rad:
        x2 += sng * stepsize
        delta += 1
        if x2 > x1:
            sng = -1
        else:
            sng = 1
        if delta > MAXITERS:
            return -1
    return delta
        
def opt_step_length(init_pos, term_pos, rad=20, stepsize=10):
    if np.abs(init_pos[0] - term_pos[0]) <= rad and np.abs(init_pos[1] - term_pos[1]) <= rad:
        return 0
    mx_1,mn_1 = np.max([init_pos[0], term_pos[0]]), np.min([init_pos[0], term_pos[0]])
    mx_2,mn_2 = np.max([init_pos[1], term_pos[1]]), np.min([init_pos[1], term_pos[1]])

    x_dist = count_steps(mx_1, mn_1, stepsize=stepsize, rad=rad) 
    y_dist = count_steps(mx_2, mn_2, stepsize=stepsize, rad=rad)
    if x_dist == -1 or y_dist == -1:
        return -1
    return x_dist + y_dist