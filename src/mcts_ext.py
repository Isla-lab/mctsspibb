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
import time
import numpy as np
import random

from src.util import generative_baseline


class ExpParams:
    def __init__(self, n_actions, n_states, exp_cost, gamma, max_depth, P, env, final_states, mask, n_sims):
        self.n_actions = n_actions
        self.n_states = n_states
        self.exp_cost = exp_cost
        self.gamma = gamma
        self.max_depth = max_depth
        self.P = P
        self.env = env
        self.terminals = final_states
        self.mask = mask
        self.n_sims = n_sims


exp_params = None


class ActionNode:
    """
    An action node, i.e. an instance representing an action in the tree (linked to a parent state node, so basically a state-action node)
    """

    def __init__(self, action, state, rollout=True):
        self.action = action

        self.visited = dict()
        for ki in exp_params.P[state][action].keys():
            self.visited[ki] = 0

        self.states = dict()
        self.n = 0
        self._value = 0
        self._total = 0
        self._rollout = rollout

    def value(self, ):
        if self.n != 0:
            return self._total / self.n
        return 0

    def simulate(self, state, depth, k):
        # (sum_p <= 0 + 0.00001) or depth > exp_params.max_depth or
        """Simulates from the action (ie state-action node)"""
        self.n += 1
        # Check if the transition is null (possibly by using the MLE) or if the depth has been reached
        sum_p = np.sum(np.array([value for value in exp_params.P[state][self.action].values()]))
        if (sum_p <= 0 + 0.00001) or depth > exp_params.max_depth:
            # print(list(exp_params.P[state][self.action].keys())[0])
            # print('STOP')
            # reward = exp_params.R[state][self.action]
            # reward = exp_params.R[state][self.action]
            reward = exp_params.env.reward_function(state, self.action)
            # In this case just return the reward
            self._value += k * reward
            return k * reward

        prob_i = np.array([value for value in exp_params.P[state][self.action].values()])
        k_i = np.array([value for value in exp_params.P[state][self.action].keys()])
        # print('state %s' %state)
        # print('action %s' %self.action)
        # print('prob_i %s' %prob_i)
        # print('k_i %s' %k_i)
        s_prime = np.random.choice(k_i, p=prob_i)

        # _rollout is a boolean: if it is true, we use a rollout strategy, otherwise we just expand all states
        # and keep simulating (very slow and hard on memory)
        if self._rollout:
            # print('if self._rollout')
            # Has not been visited before: rollout
            if self.visited[s_prime] == 0:
                # print('if-self.visited[s_prime] == 0')
                # print(sum_p)
                # print(state)
                # print(self.action)
                node = StateNodeNew(state=s_prime)
                self.states[s_prime] = node
                delayed_reward = exp_params.gamma * self.states[s_prime].rollout(s_prime, depth + 1)

            else:
                # print('else-self.visited[s_prime] == 0')
                delayed_reward = exp_params.gamma * self.states[s_prime].simulate(depth + 1)

        else:
            # print('else2-self.visited[s_prime] == 0')
            # expand if not visited
            if self.visited[s_prime] == 0:
                # print('self.visited[s_prime] == 0')
                node = StateNodeNew(state=s_prime)
                self.states[s_prime] = node

        self.visited[s_prime] += 1

        # again, if we do not rollout we need to simulate again to get the delayed reward
        if not self._rollout:
            delayed_reward = exp_params.gamma * self.states[s_prime].simulate(depth + 1)

        # actual_reward = exp_params.R[state][self.action]
        # actual_reward = exp_params.R[state][self.action]
        actual_reward = exp_params.env.reward_function(state, self.action)

        reward = (actual_reward + delayed_reward)

        self._total += k * reward

        return k * reward


class StateNode:
    def __init__(self, state=0, initial_n=0, initial_val=0, root=False):
        self.root = root
        self.state = state
        # if state not in exp_params.P.keys():
        self.pi_b_masked = generative_baseline(exp_params.env, state, dst=True)
        self.pi_b_masked[exp_params.mask[state]] = 0

        # self.ks = np.ones(exp_params.env.n_actions)
        # p_safe = 1 - sum(self.pi_b_masked)
        # for a in range(exp_params.env.n_actions):
        #     if exp_params.mask[state][a]:  # safe
        #         self.ks[a] = p_safe
        #     else:
        #         self.ks[a] = self.pi_b_masked[a]

        # Calculate safe actions
        self.non_safe_actions = np.where(~exp_params.mask[state])
        new_nonsafe = []
        for i in self.non_safe_actions[0]:
            if i in exp_params.P[state]:
                new_nonsafe.append(i)
        self.non_safe_actions = np.array(new_nonsafe)

        self.safe_actions = np.where(exp_params.mask[state])
        new_safe = []
        for i in self.safe_actions[0]:
            if i in exp_params.P[state]:
                new_safe.append(i)
        self.safe_actions = np.array(new_safe)

        self.p_nonsafe = np.sum(self.pi_b_masked)
        self.p_safe = 1 - self.p_nonsafe

        self.ks = dict()
        for a in exp_params.P[self.state].keys():
            if exp_params.mask[state][a]:  # safe
                self.ks[a] = self.p_safe
            else:
                self.ks[a] = self.pi_b_masked[a]

        self.p_pick_bootstrapped = 0
        self.initial_n = initial_n
        self.initial_val = initial_val
        self.totalN = 0
        self._oldtotal = 0
        self.n = 0
        self.actions = dict()
        # TODO: can we create only the action node we visited?
        self.create_children_all(self.state)
        # changed here UCT
        self.ns = dict()
        for i in exp_params.P[state].keys():
            self.ns[i] = self.initial_n
        # changed here UCT
        self.values = dict()
        for i in exp_params.P[state].keys():
            self.values[i] = self.initial_val

        self._value = 0  # Total value
        self.bootstrapped_values = [0 for i in range(exp_params.n_actions)]
        self.non_bootstrapped_value = 0

    def build_tree(self, budget):
        """Build the tree from this as a root by simulating budget times"""

        if budget != 0:
            timer = time.time()
            while time.time() < timer + budget:
                self.simulate(0)
        else:
            for _ in range(exp_params.n_sims):
                self.simulate(0)

        # Return the stats (needed for everything)
        return self.stats()

    def update_node_stats(self, action, reward, n=1):
        """Update node states (online update, normal way - like in UCT MCTS)"""
        self.totalN += n
        self.ns[action] += n
        self.values[action] += ((reward - self.values[action]) / self.ns[action])


class StateNodeNew(StateNode):
    # def stats(self):
    #     """Return the states: q-values, number of visits, value, old value (only UCT, no our formula)"""
    #     return [a.value() for a in self.actions.values()], self.ns, self.value_(), self.value_old()

    def stats(self):
        """Return the states: q-values, number of visits, value, old value (only UCT, no our formula)"""
        q_val = np.zeros(exp_params.env.n_actions)
        oa = collections.OrderedDict(sorted(self.actions.items()))
        for a in oa.keys():
            q_val[a] = oa[a].value()

        total_ns = np.zeros(exp_params.env.n_actions)
        on = collections.OrderedDict(sorted(self.ns.items()))
        for a in on.keys():
            total_ns[a] = on[a]

        return q_val, total_ns, self.value_(), self.value_old()

    # def create_children(self, action):
    #     """Expand"""
    #     newNode = ActionNode(action)
    #     self.actions[action] = newNode

    def create_children_all(self, state):
        """Expand all actions"""
        for a in exp_params.P[state].keys():
            self.actions[a] = ActionNode(a, state)

    def simulate(self, depth):
        """Simulate"""

        action = self.pick_action(self.p_nonsafe)
        # reward = 0
        # Boolean: is the action bootstrapped or not
        # bootstrapped = action in self.non_safe_actions[0]
        bootstrapped = action in self.non_safe_actions

        # Constant to append to the value (see formula - either pi_b(s) or p)
        # k = exp_params.ks[self.state][action]
        k = self.ks[action]

        # Simulate from the action node
        reward = self.actions[action].simulate(self.state, depth=depth, k=1)  # 1 could be k

        # Update statistics (not trivial)
        self.update_node_stats(action, reward, k, bootstrapped=bootstrapped)  # Also update state node

        return 1 * reward  # 1 could be k

    def rollout(self, state, depth):
        # act = np.sum(exp_params.pi_b_masked[state])
        pi_b_masked = generative_baseline(exp_params.env, state, dst=True)
        pi_b_masked[exp_params.mask[state]] = 0
        act = np.sum(pi_b_masked)

        # TODO change to drawing a random theta and then in case take bootstrapped or non bootstrapped
        action = self.pick_action(act, state=state)

        k = 1

        # sum_p = np.sum(exp_params.P[state][action].toarray())
        sum_p = np.sum([value for value in exp_params.P[state][action].values()])
        if (sum_p <= 0 + 0.00001) or depth > exp_params.max_depth:
            reward = k * exp_params.env.reward_function(state, action)
            return reward

        p = np.array([value for value in exp_params.P[state][action].values()])
        new_state = np.random.choice(p.shape[0], p=p)

        actual_reward = exp_params.env.reward_function(state, action)

        # RECURSIVE CALL
        delayed_reward = exp_params.gamma * self.rollout(new_state, depth + 1)

        tot = k * (actual_reward + delayed_reward)

        return tot

    # CHANGED HERE version 1:  we select randomly if any n = 0
    def pick_nonbootstrapped_action(self, state, safe):  # Never picks a non safe action

        # selection with prior knowledge (baseline) if any n = 0
        # if 0 in self.ns:
        if 0 in self.ns.values():
            # indices = [i for i, x in enumerate(self.ns) if x == 0]
            indices = []
            for i in self.ns.keys():
                if self.ns[i] == 0:
                    indices.append(i)
            probs = [1 / len(indices)] * len(indices)
            selected_action = random.choices(population=indices, weights=probs)[0]

            return selected_action

        # Must compute it
        else:
            # updated_vals = [
            #     self.actions[a].value() + exp_params.exp_cost * np.sqrt(np.log(self.totalN) / self.ns[a]) for a
            #     in self.safe_actions[0]]
            updated_vals = [
                self.actions[a].value() + exp_params.exp_cost * np.sqrt(np.log(self.totalN) / self.ns[a]) for a
                in self.safe_actions]

        # return self.safe_actions[0][np.argmax(updated_vals)]
        return self.safe_actions[np.argmax(updated_vals)]

    def pick_action(self, custom_prob, state=None):

        in_rollout = True
        if state is None:
            in_rollout = False
            state = self.state
            nonsafe = self.non_safe_actions
            safe = self.safe_actions
        else:
            nonsafe = np.where(~exp_params.mask[state])
            new_nonsafe = []
            for i in nonsafe[0]:
                if i in exp_params.P[state]:
                    new_nonsafe.append(i)
            nonsafe = np.array(new_nonsafe)

            safe = np.where(exp_params.mask[state])
            new_safe = []
            for i in safe[0]:
                if i in exp_params.P[state]:
                    new_safe.append(i)
            safe = np.array(new_safe)

        if len(nonsafe) != 0:
            # unsafe_to_pick = [a for a in range(exp_params.n_actions) if a in nonsafe[0]]  # legal
            unsafe_to_pick = [a for a in range(exp_params.n_actions) if a in nonsafe]  # legal

        else:
            unsafe_to_pick = []
        # unsafe_to_pick = [a for a in range(exp_params.n_actions) if a in nonsafe]  # legal
        # prob_baseline = np.take(exp_params.pi_b_masked[state].copy(), unsafe_to_pick)
        pi_b_masked = generative_baseline(exp_params.env, state, dst=True)
        pi_b_masked[exp_params.mask[state]] = 0
        prob_baseline = np.take(pi_b_masked, unsafe_to_pick)

        """Action selection strategy"""
        # Custom prob can be something different: we use, in general, the probability of all bootstrapped actions (1-p)
        safe_unsafe = np.random.random()
        # if len(nonsafe[0]) != 0 and safe_unsafe <= custom_prob:
        if len(nonsafe) != 0 and safe_unsafe <= custom_prob:
            # Possible non_safe actions to pick
            # to_pick = [a for a in range(0, exp_params.n_actions) if a in nonsafe[0]]  # legal
            to_pick = nonsafe  # legal

            if len(to_pick) != 0:
                # probs = 0
                # probs = np.take(exp_params.pi_b_masked[state].copy(), to_pick)
                # probs = probs / (np.sum(probs))
                probs = 0
                pi_b_masked = generative_baseline(exp_params.env, state, dst=True)
                pi_b_masked[exp_params.mask[state]] = 0
                probs = np.take(pi_b_masked, to_pick)
                probs = probs / (np.sum(probs))

                # Now the probability is 1 over all non safe actions
                r = np.random.choice(to_pick, p=probs)
                return r
        # Else, we pick from the safe actions, by using UCT
        # if len(safe[0]) != 0:
        if len(safe) != 0:
            if not in_rollout:
                # CHANGED HERE
                nonbt = self.pick_nonbootstrapped_action(state, safe)

                return nonbt
            else:
                # rc = np.random.choice(safe[0])
                rc = np.random.choice(safe)

                return rc
        # Fail safe
        fs = np.random.choice(range(exp_params.n_actions))

        return fs

    def update_node_stats(self, action, reward, k, bootstrapped=False, n=1):
        self.n += n
        self.ns[action] += n
        self._oldtotal += reward
        if bootstrapped == False:
            self.totalN += n
            self.non_bootstrapped_value += k * reward
            # Add also the normal value
        else:
            self.bootstrapped_values[action] += k * reward

    def value_(self):
        """Value of the state"""
        bval = 0
        # for a in self.non_safe_actions[0]:
        for a in self.non_safe_actions:

            if self.ns[a] != 0:
                bval += self.bootstrapped_values[a] / self.ns[a]
        nonbval = 0
        if self.totalN != 0:
            nonbval += self.non_bootstrapped_value / self.totalN

        return bval + nonbval

    def value_old(self):
        """Value as computed by the classic UCT strategy, not ours"""
        if self.n != 0:
            return self._oldtotal / self.n
        return 0


class Mcts_ext:
    """
    Class for MCTS-SPIBB. Its purpose is to hold the root of the trees, which will 'start' from an instance of StateNodeNew
    """

    def __init__(self, gamma, nb_states, nb_actions, MLE_T, env, mask, budget, n_sims,
                 max_depth=15, exploration_costant=10, states_to_sim=[],
                 final_states=[]):

        self.final_states = final_states
        self.states_to_sim = states_to_sim
        self.v = dict()
        self.v_old = dict()
        self.ns = dict()
        self.max_depth = max_depth
        self.gamma = gamma
        self.n_sims = n_sims
        self.nb_actions = nb_actions
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.P = MLE_T
        self.env = env

        # True <=> Safe
        self.mask = mask
        self.budget = budget
        self.exploration_costant = exploration_costant
        self.q_values = dict()
        self.pi = dict()

    def computeTree(self, s):
        """
        Parameters: 
            s The state for which we want to compute the tree
        """
        # Global instance to hold the experiment parameters (save memory)
        global exp_params
        exp_params = ExpParams(self.nb_actions, self.nb_states, self.exploration_costant, self.gamma,
                               self.max_depth, self.P, self.env, self.final_states, self.mask, self.n_sims)

        # Root node
        root = StateNodeNew(state=s, root=True)
        # Return all info for the state (n_visits, total_return, value etc.) building the tree from the root
        # return root.build_tree(self.n_sims)
        return root.build_tree(self.budget)

    def fit(self):

        state_sims = self.states_to_sim

        # Hold the results for each state
        results = dict()

        for s in state_sims:
            results[s] = self.computeTree(s)

        # Now results hold various statistics. We unfold them
        for s in state_sims:
            self.q_values[s] = results[s][0]
            self.ns[s] = results[s][1]
            self.v[s] = results[s][2]
            self.v_old[s] = results[s][3]
            self.pi[s] = self.new_pi(s, np.array(results[s][0]))

    # # Q must contain the tree Q values
    # def new_pi(self, s, q):
    #     """Build the new pi from the Q-values"""
    #
    #     pi = self.pi_b_masked[s].copy()
    #
    #     pi_b_masked_sum = np.sum(self.pi_b_masked[s])
    #     # pi_b is not completely non safe for this state
    #     if pi_b_masked_sum < 1 - 0.0001:
    #         #            print("new_pi() - if")
    #         safe_indices = np.where(self.mask[s])[0]
    #         pi[safe_indices[np.argmax(q[safe_indices])]] = 1 - pi_b_masked_sum
    #         # self.pi[s] = pi
    #     return pi

    # Q must contain the tree Q values
    def new_pi(self, s, q):
        """Build the new pi from the Q-values"""
        # print('state')
        # print(s)
        # print('qvalues')
        # print(q)
        pi = generative_baseline(exp_params.env, s, dst=True)
        # print('pi')
        # print(pi)
        pi[exp_params.mask[s]] = 0

        pi_sum = np.sum(pi)
        # print('pi_sum')
        # print(pi_sum)
        # pi_b is not completely non safe for this state
        if pi_sum < 1 - 0.0001:
            safe_indices = np.where(exp_params.mask[s])[0]
            # print('safe_indices')
            # print(safe_indices)
            pi[safe_indices[np.argmax(q[safe_indices])]] = 1 - pi_sum
            # self.pi[s] = pi
        return pi
