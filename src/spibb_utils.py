# authors: anonymous

import numpy as np
import time

#import pomdp_problems.tiger.tiger_problem as tp

# Sectect action based on the the action-state function with a softmax strategy
def softmax_action(Q, s):
	proba=np.exp(Q[s, :])/np.exp(Q[s, :]).sum()
	nb_actions = Q.shape[1]
	return np.random.choice(nb_actions, p=proba)


# Select the best action based on the action-state function
def best_action(Q, s):
	return np.argmax(Q[s, :])


# Compute the baseline policy, which is a softmax ovec a given function Q.
def compute_baseline(Q):
	baseline = np.exp(Q)
	norm = np.sum(baseline, axis=1).reshape(Q.shape[0], 1)
	return baseline/norm


# Prints with a time stamp
def prt(s):
	format1 = ';'.join([str(0), str(30), str(41)])
	format2 = ';'.join([str(0), str(31), str(40)])
	s1 = '\x1b[%sm %s \x1b[0m' % (format1, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	s2 = '\x1b[%sm %s \x1b[0m' % (format2, s)
	print(s1 + '  '+ s2)


# The reward function is defined on SxS, but we need it on SxA.
# This function makes the transformation based on the transition function P.
def get_reward_model(P, R):
	return np.einsum('ijk,ik->ij', P, R)


# Compute the performance of a policy given the corresponding action-state function
def compute_perf(env, gamma, Q=None, nb_trajectories=1000, max_steps=50, model=None, bootstrap=False, strategy_best=True):
	cum_rew_arr = []
	for _ in np.arange(nb_trajectories):
		isNotOver = True
		cum_rew = 0
		nb_steps = 0
		state = env.reset()
		if model != None:
			model.new_episode()
		while isNotOver and nb_steps < max_steps:
			if model != None:
				action_choice = model.predict(int(state), bootstrap)
			else:
				if strategy_best:
					action_choice = best_action(Q, int(state))
				else:
					action_choice = softmax_action(Q, int(state))
			state, reward, next_state, is_done = env.step(action_choice)
			isNotOver = not(is_done)
			cum_rew += reward*gamma**nb_steps
			nb_steps += 1
			state = next_state
		cum_rew_arr.append(cum_rew)
	expt_return = np.mean(cum_rew_arr)
	return expt_return


# Computes the monte-carlo estimation of the Q function of the behavioural policy given a batch of trajectories
def compute_q_pib_est(gamma, nb_states, nb_actions, batch):
	count_state_action = np.zeros((nb_states, nb_actions))
	q_pib_est = np.zeros((nb_states, nb_actions))
	for traj in batch:
		rev_traj = traj[::-1]
		ret = 0
		for elm in rev_traj:
			count_state_action[elm[1], elm[0]] += 1
			ret = elm[3] + gamma * ret
			q_pib_est[elm[1], elm[0]] += ret
	q_pib_est = np.divide(q_pib_est, count_state_action)
	return np.nan_to_num(q_pib_est)



# # Generates a batch of trajectories
# # Env is Garnet; however, it can simply be something with reset() and step()
# def generate_batch(nb_trajectories, transition_T, R, pi, n_states, n_actions, final_states, max_steps):
#     trajectories = []
#     for t in np.arange(nb_trajectories):
#         print(t)
#         trajectorY = []
#         state = 0 #initial state
#         is_done = False
#         for _ in np.arange(max_steps):
#             if(not is_done):
#                 action_choice = random.choices(population = range(n_actions), weights = pi[state])[0]
#                 reward = R[state][action_choice]             
#                 next_state = random.choices(population=range(n_states), weights=transition_T[state][action_choice])[0]
#                 trajectorY.append([action_choice, state, next_state, reward])
#                 state = next_state
                
#                 if(state in final_states):
#                     is_done = True
#             else:
#                 break
#             trajectories.append(trajectorY)
#     batch_traj = [val for sublist in trajectories for val in sublist]
#     return batch_traj


# Generates a batch of trajectories
# Env is Garnet; however, it can simply be something with reset() and step()
def generate_batch(nb_trajectories, env, pi, easter_egg=None, max_steps=250):
    _pi = pi
    trajectories = []
    for s in np.arange(nb_trajectories):
        nb_steps = 0
        trajectorY = []
        state = env.reset() #initial state
        is_done = False
        while nb_steps < max_steps and not is_done:
            action_choice = np.random.choice(_pi.shape[1], p=_pi[state])
            state, reward, next_state, is_done = env.step(action_choice, easter_egg)
            trajectorY.append([action_choice, state, next_state, reward])
            # trajectorY.append([np.int8(action_choice), np.int16(state), np.int16(next_state)])
            # trajectorY.append([np.int8(action_choice), np.int16(state)])

            state = next_state
            nb_steps += 1
        trajectories.append(trajectorY)
    batch_traj = [val for sublist in trajectories for val in sublist]
    return trajectories, batch_traj

"""
def generate_batch_tiger(nb_trajectories, env, pi=None, easter_egg=None, max_steps=50):
	if pi is None:
		_pi = env.baseline_pi
	else:
		_pi = pi
	trajectories = []
	for _ in np.arange(nb_trajectories):
		nb_steps = 0
		trajectorY = []
		state = env.reset() #initial state
		is_done = False
		while nb_steps < max_steps and not is_done:
			action_choice = tp.TigerAction('', index=np.random.choice(_pi.shape[1], p=_pi[state]))
			state, reward, next_state, is_done = env.step(action_choice)
			trajectorY.append([action_choice.index(), state, next_state, reward])
			state = next_state
			nb_steps += 1
		trajectories.append(trajectorY)
	batch_traj = [val for sublist in trajectories for val in sublist]
	return trajectories, batch_traj
"""
def evaluate_batch_MCTSSPIBB(nb_trajectories, env, MCTSSPIBB, gamma=0.95, easter_egg=None, max_steps=15):
	trajectories = []
	returns = []
	for _ in np.arange(nb_trajectories):
		d_return = 0
		gamma_ = 1
		nb_steps = 0
		trajectorY = []
		state = env.reset()
		is_done = False
		while nb_steps < max_steps and not is_done:
			action_choice = MCTSSPIBB.fit(state_to_sim=state, cores=0)
			state, reward, next_state, is_done = env.step(action_choice, easter_egg)
			d_return += gamma_ * reward
			gamma_ *= gamma
			trajectorY.append([action_choice, state, next_state, reward])
			state = next_state
			nb_steps += 1
		trajectories.append(trajectorY)
		returns.append(d_return)
	batch_traj = [val for sublist in trajectories for val in sublist]
	return returns, trajectories, batch_traj

class Transition():
	def __init__(self, matrix):
		self.T = matrix

	def probability(self, state, action, next_state):
		return self.T[state][action][next_state]

class Reward():
	def __init__(self, matrix):
		self.R = matrix

	def reward(self, state, action):
		return self.R[state][action]

class Policy():
	def __init__(self, matrix):
		self.pi = matrix

	def probability(self, state, action):
		return self.pi[state][action]