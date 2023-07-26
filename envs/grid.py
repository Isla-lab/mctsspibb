import os
import pickle
import random
from collections import defaultdict
from gym.envs.toy_text import discrete

import numpy as np
from gym import spaces, Env

from src.algorithms import value_iteration
from src.batch import new_batch
from src.models import construct_model_from_batch
from src.policies import RandomEnvPolicy, SoftMaxPolicy

"""
    Grid world from Laroche, R., & Trichelair, P. (2017). Safe Policy Improvement with Baseline Bootstrapping. (http://arxiv.org/abs/1712.06924)
"""


class GridEnv(discrete.DiscreteEnv):
    readable_actions = ["UP   ", "RIGHT", "DOWN ", "LEFT"]
    effect_prob = [
        ([(0, 1), (1, 0), (0, -1), (-1, 0)], 0.75),  # suc
        ([(0, -1), (-1, 0), (0, 1), (1, 0)], 0.05),  # fail_opposite
        ([(1, 0), (0, 1), (-1, 0), (0, -1)], 0.10),  # fail_side1
        ([(-1, 0), (0, -1), (1, 0), (0, 1)], 0.10),  # fail_side2
    ]

    def __init__(self, initial_states=None):
        nS = 25
        nA = 4
        isd = np.zeros(nS)
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        if initial_states is None:
            isd[self.encode(0, 0)] = 1
        else:
            for s in initial_states:
                isd[self.encode(*s)] = 1./len(initial_states)
        self.goal_state = self.encode(4, 4)

        for s in range(nS):
            for a in range(nA):
                x, y = self.decode(s)
                next_states = defaultdict(float)

                for effect, p in self.effect_prob:
                    delta_x, delta_y = effect[a]
                    new_x, new_y = GridEnv.new_state(x, y, delta_x, delta_y)
                    next_states[self.encode(new_x, new_y)] += p
                for ns, p in next_states.items():
                    reward = 0
                    done = False
                    if s == self.goal_state:
                        reward = 0
                    elif s == ns:
                        reward = -10
                    elif ns == self.goal_state:
                        reward = 100
                        done = True
                    P[s][a].append((p, ns, reward, done))
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    @staticmethod
    def new_state(x, y, delta_x, delta_y):
        if not GridEnv._hit_internal_wall(x, y, delta_x):
            return GridEnv._new_position(x, y, delta_x, delta_y)
        else:
            return x, y

    @staticmethod
    def _new_position(x, y, delta_x, delta_y):
        new_x = max(min(x + delta_x, 4), 0)
        new_y = max(min(y + delta_y, 4), 0)
        return new_x, new_y

    @staticmethod
    def _hit_internal_wall(x, y, delta_x):
        return (x == 2 and y in range(3) and delta_x == 1
                or x == 3 and y in range(3) and delta_x == -1
                or x == 3 and y in range(2, 4) and delta_x == 1
                or x == 4 and y in range(2, 4) and delta_x == -1)

    @staticmethod
    def print_policy(policy):
        for j in reversed(range(5)):
            line = ""
            for i in range(5):
                line += GridEnv.readable_actions[policy((i, j))] + "\t"
            print(line)

    @staticmethod
    def get_policy_from_q_values_file(file_name):
        loaded_q = np.load(file_name)
        mdp = construct_model_from_batch([], GridEnv(), behavior_policy=None)
        q_values = np.zeros(shape=(25, 4))
        for state_id in range(25):
            for a in range(4):
                q_values[state_id, a] = loaded_q[state_id][a]
                # print(x, y, GridEnv.readable_actions[a], Q_baseline[state_id][a], f[(x, y), my_actions[a]])

        return SoftMaxPolicy(mdp=mdp, q_values=q_values, t=1)

    @staticmethod
    def encode(x, y):
        return x + 5 * y

    @staticmethod
    def decode(state_id):
        x = state_id % 5
        y = state_id // 5
        return x, y
