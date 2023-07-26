"""
This file is part of MCTS-SPIBB.

MCTS-SPIBB is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar.
If not, see <https://www.gnu.org/licenses/>.

MCTS-SPIBB authors kindly ask to cite the paper "Scalable Safe Policy Improvement via Monte Carlo Tree Search" if the
code is used in other projects.
"""

import collections

import gym
import pickle
import random
import numpy as np

from src import models, garnets, spibb
from src.algorithms import value_iteration
from src.batch import new_batch
from src.policies import SoftMaxPolicy, RandomEnvPolicy


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def parse_json(path):
    import json
    params = read_file(path)
    return json.loads(params)


def get_configs(d):
    for k, v in d.items():
        if type(v) is list and len(v) > 0:
            result = []
            for x in v:
                new_dict = dict(d)
                new_dict[k] = x
                result.append(new_dict)

            return result
    return [d]


def get_configs_from_dicts(list_of_dicts):
    result = []
    for d in list_of_dicts:
        result += get_configs(d)
    if len(result) == len(list_of_dicts):
        return result
    else:
        return get_configs_from_dicts(result)


def save_object(obj, filename):
    with open(filename, 'wb') as out:  # Overwrites any existing file.
        pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)


def compute_mask(matrix_state_action_visits, N_wedge):
    return matrix_state_action_visits > N_wedge


def compute_compressed_mask(n, n_wedge):
    mask = dict()
    os = np.array(list(n.keys()))
    ordered_states = np.sort(os)
    for state in ordered_states:
        if state not in mask:
            mask[state] = np.zeros(len(n[0]), dtype=bool)
        oa = np.array(list(n[state].keys()))
        ordered_actions = np.sort(oa)
        for action in ordered_actions:
            if n[state][action] > n_wedge:
                mask[state][action] = True
    return mask


def decode(i, n_machines):
    out = bin(i)[2:].rjust(n_machines, '0')
    return map(int, reversed(out))


def ul(x):
    return "1" if x else "0"


def policy_random_baseline_generator(n_states, n_actions, p=0.4):
    pi_b_rand = np.zeros(shape=(n_states, n_actions))

    for s in range(n_states):
        pi_b_rand[s, random.randint(0, n_actions - 1)] = p
        indices = [i for i, x in enumerate(pi_b_rand[s]) if x == 0]
        probs = [(1 - p) / len(indices)] * len(indices)

        for idx in indices:
            pi_b_rand[s, idx] = probs[0]

    return pi_b_rand


def generative_baseline(env, state, p=0.70, rho=0, p_shuffle=0, dst=False):
    state = np.array(list(env.decode(state)))
    machines_on = np.where(state == 1)[0]
    machines_off = np.where(state == 0)[0]
    pi = np.zeros(env.n_actions)

    if machines_on.size == env.n_machines:
        pi[env.n_actions - 1] = p
        pi[:env.n_actions - 1] = (1 - p) / (env.n_actions - 1)

    elif machines_off.size == env.n_machines:
        pi[0] = p
        indices = [i for i, x in enumerate(pi) if x == 0]
        probs = [(1 - p) / (len(pi) - 1)] * len(indices)
        pi[indices] = probs

    else:
        for s in machines_on:
            if (s + 1) % (env.n_actions - 1) in machines_off:
                pi[s + 1] = p
                indices = [i for i, x in enumerate(pi) if i != s + 1]
                probs = [(1 - p) / len(indices)] * len(indices)
                pi[indices] = probs
                break
            elif (s - 1) % (env.n_actions - 1) in machines_off:
                pi[s - 1] = p
                indices = [i for i, x in enumerate(pi) if i != s - 1]
                probs = [(1 - p) / len(indices)] * len(indices)
                pi[indices] = probs
                break
            continue
    pi /= pi.sum()

    if not dst:
        return np.random.choice(pi.shape[0], p=pi)
    else:
        return pi


def baseline_policy_generator(n_machines, n_states, n_actions, p=0.85, rho=0, p_shuffle=0):
    encode = []
    for i in range(n_states):
        machines_on = list(decode(i, n_machines))
        encode.append([ul(x) for x in machines_on])

    pi = np.zeros((n_states, n_actions))

    # First state
    pi[0, random.randint(0, n_actions - 2)] = p
    indices = [i for i, x in enumerate(pi[0]) if x == 0]
    probs = [(1 - p) / len(indices)] * len(indices)
    pi[0, indices] = probs

    # Last state
    pi[n_states - 1, n_actions - 1] = p
    pi[n_states - 1, : n_actions - 1] = (1 - p) / (n_actions - 1)

    # Other states
    index = 1
    for s in encode[1:n_states - 1]:

        machines_1 = [i for i, x in enumerate(s) if "1" in x]
        random.shuffle(machines_1)

        for s1 in machines_1:

            left_or_right = random.randint(0, 1)

            if (left_or_right == 0):
                s1_lr = (s1 - 1) % (n_actions - 1)
                if "0" in s[s1_lr]:
                    pi[index, s1_lr] = p
                    indices = [i for i, x in enumerate(pi[index]) if i != s1_lr]
                    probs = [(1 - p) / len(indices)] * len(indices)
                    pi[index, indices] = probs
                    break

                s1_lr = (s1 + 1) % (n_actions - 1)
                if "0" in s[s1_lr]:
                    pi[index, s1_lr] = p
                    indices = [i for i, x in enumerate(pi[index]) if i != s1_lr]
                    probs = [(1 - p) / len(indices)] * len(indices)
                    pi[index, indices] = probs
                    break

                continue
            else:
                s1_lr = (s1 + 1) % (n_actions - 1)
                if "0" in s[s1_lr]:
                    pi[index, s1_lr] = p
                    indices = [i for i, x in enumerate(pi[index]) if i != s1_lr]
                    probs = [(1 - p) / len(indices)] * len(indices)
                    pi[index, indices] = probs
                    break

                s1_lr = (s1 - 1) % (n_actions - 1)
                if "0" in s[s1_lr]:
                    pi[index, s1_lr] = p
                    indices = [i for i, x in enumerate(pi[index]) if i != s1_lr]
                    probs = [(1 - p) / len(indices)] * len(indices)
                    pi[index, indices] = probs
                    break

                continue

        index += 1
    if rho != 0:
        pi_b_rand = policy_random_baseline_generator(n_states, n_actions, p=0.4)
        # Percentage of errors in selection of actions
        if p_shuffle != 0:
            n_states_to_shuffle = int(n_states * p_shuffle)
            states_to_shuffle = random.sample(range(n_states), n_states_to_shuffle)

            for s in states_to_shuffle:
                random.shuffle(pi[s])

        # A new policy with noise
        return rho * pi + (1 - rho) * pi_b_rand

        # A new policy without noise and errors
    return pi


def new_episode(env, episode_length_bound, policy):
    state = env.reset()
    episode = list()
    for t in range(episode_length_bound):
        action = np.random.choice(policy.shape[1], p=policy[state])
        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward, next_state))
        if done:
            break
        state = next_state
    return episode


def new_episode_with_generative_baseline(env, episode_length_bound):
    state = env.reset()
    episode = list()
    for t in range(episode_length_bound):
        action = generative_baseline(env, state)
        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward, next_state))
        if done:
            break
        state = next_state
    return episode


def get_random_policy_and_env(baseline_policy, env_id, discount_factor, softmax_temperature,
                              baseline_policy_path="", seed=123):
    wrapped_env = gym.envs.make(env_id)
    wrapped_env.seed(seed)
    env = wrapped_env.env

    behavior_policy = get_behavior_policy(env, env_id, baseline_policy, discount_factor,
                                          softmax_temperature, baseline_policy_path)

    return behavior_policy, env


def load_env(env_id, seed=123):
    print('Compute the environment')
    wrapped_env = gym.envs.make(env_id)
    wrapped_env.seed(seed)
    env = wrapped_env.env
    save_object(env, 'out/env_%s.pkl' % env_id)
    print('Environment is created')
    return env


def get_behavior_policy(env, env_id, baseline_policy, discount_factor, softmax_temperature, baseline_policy_path):
    mdp, optimal_q_values = get_optimal_q_values(env, env_id, discount_factor)
    behavior_policy = SoftMaxPolicy(mdp=mdp, q_values=optimal_q_values, t=softmax_temperature)

    return behavior_policy


def get_optimal_q_values(env, env_name, discount_factor):
    random_policy = RandomEnvPolicy(env)

    batch = new_batch(env, size=10000, policy=random_policy, episode_length_bound=200)

    mdp = models.construct_model_from_batch(batch, env, behavior_policy=random_policy,
                                            discount_factor=discount_factor)

    optimal_q_values = value_iteration(mdp, max_iterations=200)

    return mdp, optimal_q_values


def policy_evaluation(pi, R, T, gamma):
    """
    Evaluate policy by taking the inverse
    Args:
      pi: policy, array of shape |S| x |A|
      r: the true rewards, array of shape |S| x |A|
      p: the true state transition probabilities, array of shape |S| x |A| x |S|
    Return:
      v: 1D array with updated state values
    """
    # Rewards according to policy: Hadamard product and row-wise sum
    r_pi = np.einsum('ij,ij->i', pi, R)

    # Policy-weighted transitions:
    # multiply p by pi by broadcasting pi, then sum second axis
    # result is an array of shape |S| x |S|
    p_pi = np.einsum('ijk, ij->ik', T, pi)
    v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - gamma * p_pi)), r_pi)
    return v, R + gamma * np.einsum('i, jki->jk', v, T)


def load_env_and_policies_gridworld(size, discount_factor):
    path = 'scripts/configs/gridworld_size_%s/' % size
    real_T = np.load(path + 'real_T_size_%s.npy' % size)
    discount_factor = discount_factor
    nb_states = size * size
    nb_actions = 4
    x_max = size
    y_max = size
    x_end = 0
    y_end = 3
    s_end = y_end * y_max + x_end

    final_states = [s_end, nb_states - 1]

    garnet = garnets.Garnets(nb_states, nb_actions, 1, self_transitions=0)
    garnet.transition_function = real_T
    R = np.load(path + 'real_R_size_%s.npy' % size)
    mask_0 = np.full((nb_states, nb_actions), True)
    optimal = spibb.spibb(discount_factor, nb_states, nb_actions, mask_0, mask_0, real_T, R, 'default')
    optimal.fit()

    baseline = pickle.load(open(path + 'baseline_size_%s.pkl' % size, 'rb'))

    return real_T, R, optimal, baseline, garnet, final_states, path

def load_env_and_policies_sysadmin(size, discount_factor):
    path = 'scripts/configs/sysadmin_%s/' % size
    real_T = np.load(path + 'Sysadmin_%s_T.npy' % size)
    discount_factor = discount_factor
    nb_states = 2**size
    nb_actions = size + 1
    final_states = [nb_states - 1]
    garnet = garnets.Garnets(nb_states, nb_actions, 1, self_transitions=0)
    garnet.transition_function = real_T
    R = np.load(path + 'Sysadmin_%s_R.npy' % size)
    mask_0 = np.full((nb_states, nb_actions), True)
    optimal = spibb.spibb(discount_factor, nb_states, nb_actions, mask_0, mask_0, real_T, R, 'default')
    optimal.fit()

    baseline = np.load(path + 'Sysadmin_%s_baseline.npy' % size)

    return real_T, R, optimal, baseline, garnet, final_states, path