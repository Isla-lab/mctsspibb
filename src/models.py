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

import logging
import importlib
import pickle

import numpy as np

from abc import abstractmethod, ABC

from src import policies


class EstimatedMDP(ABC):
    representation = ''

    def __init__(self, env, behavior_policy, policy_class=policies.GreedyPolicy,
                 discount_factor=0.95, config=None):
        self.env = env
        self.behavior_policy = behavior_policy
        if callable(policy_class):
            self.SafePolicy = policy_class
        else:
            self.SafePolicy = load(str(policy_class))
        if config is None:
            self.config = dict()
        else:
            self.config = config
        self.discount_factor = discount_factor
        self.actions = range(self.env.nA)
        self.states = range(self.env.nS)
        self.cache_successors = [[None for _ in self.actions] for _ in range(self.env.nS + 1)]
        self.cache_bootstrapped = [[None for _ in self.actions] for _ in range(self.env.nS + 1)]
        self.transition_function_table = np.zeros(shape=(self.env.nS, self.env.nA, self.env.nS), dtype=np.float32)
        logging.info("{} instantiated".format(self))

    def __str__(self):
        return self.representation + " " + self.SafePolicy.name

    @classmethod
    def from_mdp(cls, other):
        new_mdp = cls(other.env, other.behavior_policy)
        new_mdp.copy_from(other)
        return new_mdp

    @abstractmethod
    def copy_from(self, other):
        self.discount_factor = other.discount_factor
        self.config = other.config

    def process_batch(self, batch):
        self.count(batch)
        self.compute_transition_function()

    @abstractmethod
    def count(self, batch):
        pass

    def compute_transition_function(self):
        for s in self.states:
            for a in self.actions:
                next_state_distribution = np.zeros(len(self.states))
                for p, ns in self.successors(s, a):
                    next_state_distribution[ns] = p
                self.transition_function_table[s, a] = next_state_distribution

    @abstractmethod
    def transition_function(self, s, a, ns):
        return 0

    def get_greedy_policy(self, q_values):
        return self.SafePolicy(mdp=self, q_values=q_values)

    def get_safe_policy(self, q_values):
        return self.SafePolicy(mdp=self, q_values=q_values)

    def reward_function(self, s, a):
        successors = self.env.P[s][a]
        expected_reward = sum(r * p for p, _, r, _ in successors)
        return expected_reward

    def next_state_dist(self, s, a):
        return self.transition_function_table[s, a:a + 1, :].T

    def successors(self, s, a):
        if self.cache_successors[s][a] is None:
            self.cache_successors[s][a] = self.get_successors_of(s, a)
        return list(self.cache_successors[s][a])

    @abstractmethod
    def get_successors_of(self, s, a):
        return list()

    def structure_error(self):
        return {
            "parents_not_added": 0,
            "non_parents_added": 0
        }

    def bootstrapped(self, s, a):
        if self.cache_bootstrapped[s][a] is None:
            self.cache_bootstrapped[s][a] = self.bootstrap(s, a)
        return self.cache_bootstrapped[s][a]

    @abstractmethod
    def bootstrap(self, s, a):
        pass

    def set_min_observations(self, m):
        self.config["flat_min_obs"] = m
        self.cache_bootstrapped = [[None for _ in self.actions] for _ in self.states]

    def bootstrapped_actions(self, s):
        return [a for a in self.actions if self.bootstrapped(s, a)]

    def non_bootstrapped_actions(self, s):
        return [a for a in self.actions if not self.bootstrapped(s, a)]

    def bootstrapped_state_action_pairs(self):
        result = list()
        for s in self.states:
            for a in self.actions:
                result.append(self.bootstrapped(s, a))
        return result

    def minimum_number_of_observations(self):
        # TODO: possibly compute this value according to parameters epsilon(model_precision) and delta(confidence_level)
        return self.config.get("flat_min_obs", 10)

    def divergence_transition_function(self):
        divergences = []
        for s in self.states:
            for a in self.actions:
                divergences.append(self.get_divergence_transition(s, a))
        return divergences

    def get_divergence_transition(self, s, a):
        res = 0
        i = 0
        j = 0
        while i < len(self.successors(s, a)) and j < len(self.env.P[s][a]):
            pi, si = self.successors(s, a)[i]
            (pj, sj, _, _) = self.env.P[s][a][j]
            if si == sj:
                res += abs(pi - pj)
                i += 1
                j += 1
            elif si < sj:
                res += pi
                i += 1
            else:  # sj > si
                res += pj
                j += 1
        while i < len(self.successors(s, a)):
            res += self.successors(s, a)[i][0]
            i += 1
        while j < len(self.env.P[s][a]):
            res += self.env.P[s][a][j][0]
            j += 1
        return res

    def dump_transition_function(self):
        divergences = []
        for s in self.states:
            for a in self.actions:
                print(s, a, "\t ".join("{} {}".format(ns, p) for p, ns in self.successors(s, a)))
        return np.mean(divergences), np.std(divergences)


class EstimatedEnumeratedMDP(EstimatedMDP):
    representation = "Flat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nA = self.env.nA
        nS = self.env.nS
        self.is_value_based_policy = self.SafePolicy in [policies.ValueBasedSafeGreedyPolicy,
                                                         policies.ValueBasedInfinity]
        if self.is_value_based_policy:
            self.states = range(nS + 1)
            self.dummy_state = nS
            nS = nS + 1
            self.transition_function_table = np.zeros(shape=(nS, nA, nS), dtype=np.float32)
        self.t = np.zeros(shape=(nS, nA, nS), dtype=np.int)  # transitions counter
        self.n = np.zeros(shape=(nS, nA), dtype=np.int)  # state-action counter
        self.q = np.zeros(shape=(nS, nA), dtype=np.float)  # q-value sum

    def copy_from(self, other):
        super().copy_from(other)
        for s in other.states:
            for a in other.actions:
                for ns in other.states:
                    self.t[s, a, ns] = other.t[s, a, ns]
                self.n[s, a] = other.n[s, a]
                self.q[s, a] = other.q[s, a]

    def transition_function(self, s, a, ns):
        if self.is_value_based_policy:
            if s == self.dummy_state:
                return 0
            if self.bootstrapped(s, a):
                if ns == self.dummy_state:
                    return 1
                else:
                    return 0
            if ns == self.dummy_state:
                return 0
        if self.n[s, a] == 0:
            return 0
        return self.t[s, a, ns] / self.n[s, a]

    def get_successors_of(self, s, a):
        result = list()
        for ns in self.states:  # self.t[s, a].nonzero()[0]:
            p = self.transition_function(s, a, ns)
            if p > 0:
                result.append((p, ns))
        return result

    def reward_function(self, s, a):
        if self.is_value_based_policy:
            if self.bootstrapped(s, a):
                if self.n[s, a] == 0:
                    # assume q-value of state-action pairs never seen before is zero
                    return 0
                else:
                    return self.q[s, a] / self.n[s, a]
        return super().reward_function(s, a)

    def q_value(self, s, a):
        if self.n[s, a] > 0:
            return self.q[s, a] / self.n[s, a]
        else:
            return 0

    def count(self, batch):
        for episode in batch:
            past_state_actions = list()
            for time, (state, action, reward, next_state) in enumerate(episode):
                past_state_actions.append((state, action, time))
                self.n[state, action] += 1
                self.t[state, action, next_state] += 1

                for (past_state, past_action, t) in past_state_actions:
                    relative_time = time - t
                    self.q[past_state, past_action] += self.discount_factor ** relative_time * reward

    def get_soft_max_policy(self, q_values, t=1):
        return policies.SoftMaxPolicy(mdp=self, q_values=q_values, t=t)

    def bootstrap(self, s, a):
        if self.SafePolicy == policies.GreedyPolicy:
            return False
        else:
            return self.n[s, a] < self.minimum_number_of_observations()

    def get_greedy_policy(self, q_values):
        if self.is_value_based_policy:
            return policies.GreedyPolicy(mdp=self, q_values=q_values)
        return self.SafePolicy(mdp=self, q_values=q_values)


def construct_model_from_batch(batch,
                               env,
                               behavior_policy,
                               representation=EstimatedEnumeratedMDP,
                               safety_strategy=policies.GreedyPolicy,
                               structure_type=None,
                               discount_factor=0.95,
                               config=None,
                               other_mdp=None):
    if config is None:
        config = dict()
    if not callable(representation):
        representation = load(str(representation))

    config['structure_type'] = structure_type
    mdp = representation(env, behavior_policy, safety_strategy, discount_factor=discount_factor, config=config)
    if other_mdp is None:
        mdp.process_batch(batch)
    else:
        mdp.copy_from(other_mdp)

    return mdp


def save_object(obj, filename):
    with open(filename, 'wb') as out:  # Overwrites any existing file.
        pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)


def get_baseline_policy(batch,
                        env,
                        env_id,
                        behavior_policy,
                        representation=EstimatedEnumeratedMDP,
                        safety_strategy=policies.GreedyPolicy,
                        structure_type=None,
                        discount_factor=0.95,
                        config=None,
                        other_mdp=None):
    if config is None:
        config = dict()
    if not callable(representation):
        representation = load(str(representation))

    config['structure_type'] = structure_type
    mdp = representation(env, behavior_policy, safety_strategy, discount_factor=discount_factor, config=config)
    if other_mdp is None:
        mdp.process_batch(batch)
    else:
        mdp.copy_from(other_mdp)

    save_object(mdp.behavior_policy.full_distribution(), 'out/SysAdmin/baseline_%s.pkl' % env_id)
    save_object(mdp.transition_function_table, 'out/SysAdmin/mle/MLE_T_%s.pkl' % env_id)
    save_object(mdp.n, 'out/SysAdmin/q/matrix_of_visits_%s.pkl' % env_id)
    save_object(mdp.behavior_policy.q_values, 'out/q_values_%s.pkl' % env_id)
    save_object(env, 'out/SysAdmin/env_%s.pkl' % env_id)
    save_object(mdp, 'out/SysAdmin/mdp_%s.pkl' % env_id)


def load(entry_point):
    mod_name, attr_name = entry_point.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def get_data(batch,
             env,
             env_id,
             behavior_policy,
             representation=EstimatedEnumeratedMDP,
             safety_strategy=policies.GreedyPolicy,
             structure_type=None,
             discount_factor=0.95,
             config=None,
             other_mdp=None):
    if config is None:
        config = dict()
    if not callable(representation):
        representation = load(str(representation))

    config['structure_type'] = structure_type
    mdp = representation(env, behavior_policy, safety_strategy, discount_factor=discount_factor, config=config)
    if other_mdp is None:
        mdp.process_batch(batch)
    else:
        mdp.copy_from(other_mdp)

    return mdp.behavior_policy.full_distribution(), mdp

