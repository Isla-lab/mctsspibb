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

from abc import abstractmethod, ABC
from functools import lru_cache

import numpy as np


def arg_max(q_values, state, actions):
    greedy_action = actions[0]
    for a in actions:
        if q_values[state, a] > q_values[state, greedy_action]:
            greedy_action = a
    return greedy_action


def second_best(values):
    bests = (0, 0)
    for i, v in enumerate(values):
        if v > values[bests[0]]:
            bests = i, bests[0]
        elif v > values[bests[1]]:
            bests = bests[0], i
    return bests[1]


def soft_max(w, t=1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


def sample_discrete_distribution(a_list, dist):
    samples = np.random.choice(a_list, 1, p=dist)
    return samples[0]


def dicts_to_csv(data, file_name):
    with open(file_name, 'a') as f:
        w = csv.DictWriter(f, sorted(data[0].keys()))
        if f.tell() == 0:
            w.writeheader()
        for entry in data:
            w.writerow(entry)


def product(*args, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield prod


def cross_product_of_cpts(list_of_cpts, env):
    results = []
    if not list_of_cpts:
        return results
    for l in product(*list_of_cpts):
        next_state = []
        total_p = 1
        for p, v in l:
            total_p *= p
            next_state.append(v)
        results.append((total_p, env.encode(*next_state)))
    return sorted(results, key=lambda x: x[1])



class Policy(ABC):
    @abstractmethod
    def __call__(self, state):
        pass

    def reset(self):
        """
        q-bootstrapped policy alter its behavior during execution,
        after each execution the policy should be altered to its original state using reset
        """
        pass

    @abstractmethod
    def distribution(self, state):
        pass

    @abstractmethod
    def full_distribution(self):
        """
        returns an array of distribution over actions for each state
        """
        pass

    def save(self, path):
        np.savetxt(fname=path, X=self.full_distribution(), delimiter=',')

    def dump(self, file_name, pretty=False):
        if pretty:
            self.pretty_dump(file_name)
        else:
            self.save(file_name)

    def pretty_dump(self, file_name):
        raise NotImplementedError()


class RandomEnvPolicy(Policy):

    def __init__(self, env):
        self.env = env

    def __call__(self, state):
        return self.env.action_space.sample()

    def distribution(self, state):
        n_actions = self.env.nA
        return np.full(shape=(n_actions,), fill_value=1. / n_actions, dtype=float)

    def full_distribution(self):
        n_actions = self.env.nA
        n_states = self.env.nS
        return np.full(shape=(n_actions, n_states), fill_value=1. / n_actions, dtype=float)

    def pretty_dump(self, file_name):
        with open(file_name, 'w') as f:
            for s in range(self.env.nS):
                dist = self.distribution(s)
                text_state = "{};".format(list(self.env.decode(s)))
                text_dist = ";".join("{:d};{:.2f}".format(a, p) for a, p in enumerate(dist) if p > 0)
                f.write(text_state + text_dist + "\n")


class DeterministicPolicy(Policy):
    """
    policies based on MDPs
    """
    @abstractmethod
    def __call__(self, state):
        pass

    def __init__(self, mdp, q_values):
        self.mdp = mdp
        self.q_values = q_values

    def distribution(self, state):
        # returns a vector where all the probability mass is attributed to the greedy action
        result = np.zeros(self.mdp.env.nA)
        chosen_action = self(state)
        result[chosen_action] = 1
        return result

    def full_distribution(self):
        result = np.zeros((self.mdp.env.nS, self.mdp.env.nA))
        for s in range(self.mdp.env.nS):
            chosen_action = self(s)
            result[s, chosen_action] = 1
        return result

    def pretty_dump(self, file_name):
        with open(file_name, 'w') as f:
            for s in self.mdp.states:
                dist = self.distribution(s)
                text_state = "{};".format(list(self.mdp.env.decode(s)))
                text_dist = ";".join("{:d};{:.2f}".format(a, p) for a, p in enumerate(dist) if p > 0)
                f.write(text_state + text_dist + "\n")


class GreedyPolicy(DeterministicPolicy):
    def __call__(self, state):
        return np.argmax(self.q_values[state])

    name = "Basic RL"

    def full_distribution(self):
        result = np.zeros((len(self.mdp.states), len(self.mdp.actions)))
        greedy_action_indices = self.q_values.argmax(axis=1)
        result[np.arange(len(self.mdp.states)), greedy_action_indices] = 1
        return result


class SecondBestPolicy(DeterministicPolicy):
    def __call__(self, state):
        return second_best(self.q_values[state])


class StochasticPolicy(DeterministicPolicy):
    """policies based on a distribution"""
    def __call__(self, state):
        dist = self.distribution(state)
        return sample_discrete_distribution(self.mdp.actions, dist)

    def full_distribution(self):
        result = np.zeros((self.mdp.env.nS, self.mdp.env.nA))
        for s in self.mdp.states:
            result[s] = self.distribution(s)
        return result


class SoftMaxPolicy(StochasticPolicy):
    def __init__(self, *args, t=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t

    @lru_cache(maxsize=None)
    def _soft_max_policy_dist(self, state):
        w = np.array([self.q_values[state, a] for a in self.mdp.actions], dtype=float)
        return soft_max(w, self.t)

    distribution = _soft_max_policy_dist


class ValueBasedSafeGreedyPolicy(DeterministicPolicy):
    name = "$Q^{\pi_b}$-SPIBB-1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrapped_action_state = False

    def __call__(self, state):
        if self.bootstrapped_action_state:
            return self.mdp.behavior_policy(state)

        action = arg_max(self.q_values, state, self.mdp.actions)
        if self.mdp.bootstrapped(state, action):
            self.bootstrapped_action_state = True
        return action

    def reset(self):
        self.bootstrapped_action_state = False


class ValueBasedInfinity(GreedyPolicy):
    name = "$Q^{\pi_b}$-SPIBB-$\infty$"


class PolicyBasedSafeGreedyPolicy(StochasticPolicy):
    @lru_cache(maxsize=None)
    def distribution(self, state):
        dist = self._get_dist_of_bootstrapped_actions(state)
        non_bootstrapped_actions = self.mdp.non_bootstrapped_actions(state)
        if non_bootstrapped_actions:
            non_bootstrapped_actions_q_values = {(state, a): self.q_values[state, a] for a in non_bootstrapped_actions}
            greedy_action = arg_max(non_bootstrapped_actions_q_values, state, non_bootstrapped_actions)
            dist[self.mdp.actions.index(greedy_action)] = abs(1 - sum(dist))
        return dist

    def _get_dist_of_bootstrapped_actions(self, state):
        dist = np.zeros(len(self.mdp.actions))
        behavior_dist = self.mdp.behavior_policy.distribution(state)
        for action in self.mdp.bootstrapped_actions(state):
            ind = self.mdp.actions.index(action)
            dist[ind] = behavior_dist[ind]
        return dist

    name = "$\Pi_b$-SPIBB"


class PolicyBased0SafeGreedyPolicy(PolicyBasedSafeGreedyPolicy):
    @lru_cache(maxsize=None)
    def distribution(self, state):
        dist = np.zeros(len(self.mdp.actions))
        non_bootstrapped_actions = self.mdp.non_bootstrapped_actions(state)
        if non_bootstrapped_actions:
            non_bootstrapped_actions_q_values = {(state, a): self.q_values[state, a] for a in non_bootstrapped_actions}
            greedy_action = arg_max(non_bootstrapped_actions_q_values, state, non_bootstrapped_actions)
            dist[self.mdp.actions.index(greedy_action)] = 1
        else:
            dist = self._get_dist_of_bootstrapped_actions(state)

        return dist

    name = "$\Pi_0$-SPIBB"


class PolicyBasedLessBSafeGreedyPolicy(StochasticPolicy):
    @lru_cache(maxsize=None)
    def distribution(self, state):
        dist = np.zeros(len(self.mdp.actions))
        ordered_actions = self._get_ordered_actions(state)
        mass_att = 0
        for action in ordered_actions:
            action_ind = self.mdp.actions.index(action)

            if self.mdp.bootstrapped(state, action):
                if mass_att + self.mdp.behavior_policy.distribution(state)[action_ind] > 1:
                    dist[action_ind] = 1 - mass_att
                    mass_att += dist[action_ind]
                    break
                else:
                    dist[action_ind] = self.mdp.behavior_policy.distribution(state)[action_ind]
                    mass_att += self.mdp.behavior_policy.distribution(state)[action_ind]
            else:
                dist[action_ind] = 1 - mass_att
                mass_att += dist[action_ind]
                break

        # TODO remove this hack, find why sometimes dist has negative values
        dist = np.abs(dist)
        # assert abs(mass_att - 1) < 0.00000001
        # assert abs(dist.sum() - 1) < 0.000000001
        return dist

    def _get_ordered_actions(self, state):
        return sorted(self.mdp.actions, key=lambda a: self.q_values[state, a], reverse=True)

    name = "$\Pi_{\leq b}$-SPIBB"


class PolicyFromFile(Policy):
    def __init__(self, path):
        self.policy = self.load(path)
        self.n_states, self.n_actions = self.policy.shape
        assert self.policy.dtype == float

    def __call__(self, state):
        dist = self.distribution(state)
        return sample_discrete_distribution(range(self.n_actions), dist)

    def distribution(self, state):
        return self.policy[state]

    def full_distribution(self):
        return self.policy

    @staticmethod
    def load(path):
        return np.loadtxt(fname=path, dtype=float, delimiter=',', ndmin=2)
