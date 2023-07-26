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

import time
import numpy as np
import multiprocessing
# from random import shuffle
import random
import pandas as pd

uniform = True


class ExpParams():
    def __init__(self, n_actions, n_states, exp_cost, gamma, max_depth, P, R, final_states,
                 mask, pi_b_masked, n_sims, state_node_type='uniform', fast_ucb=None, fast_ucb_limit=0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.exp_cost = exp_cost
        self.gamma = gamma
        self.max_depth = max_depth
        self.P = P
        self.R = R
        self.mask = mask
        self.n_sims = n_sims
        self.pi_b_masked = pi_b_masked
        self.fast_ucb = fast_ucb
        self.fast_ucb_limit = fast_ucb_limit
        self.state_node_type = state_node_type
        if fast_ucb is not None:
            indices = np.indices((fast_ucb_limit, fast_ucb_limit))
            self.fast_ucb = np.log(indices[1])
            self.fast_ucb /= indices[0]
            self.fast_ucb = exp_cost * np.sqrt(self.fast_ucb)
            # [0][0] and [0][1] are wrong: we fix them
            self.fast_ucb[0][0] = float('+inf')
            self.fast_ucb[0][1] = float('+inf')
        self.terminals = final_states
        self.legal_actions = np.zeros((n_states, n_actions))
        for s in range(0, n_states):
            for a in range(0, n_actions):
                if np.sum(P[s][a]) > 1e-4 or s in self.terminals:
                    self.legal_actions[s][a] = 1
                else:
                    if fast_ucb is not None:
                        self.fast_ucb[s][a] = float('-inf')
        self.ks = np.ones((n_states, n_actions))
        for s in range(0, n_states):
            p_safe = 1 - sum(pi_b_masked[s])
            for a in range(0, n_actions):
                if mask[s][a] == True:  # safe
                    self.ks[s][a] = p_safe
                else:
                    self.ks[s][a] = pi_b_masked[s][a]


exp_params = None


class ActionNode():
    """
    An action node, i.e. an instance representing an action in the tree (linked to a parent state node, so basically a state-action node)
    """

    def __init__(self, action, state, rollout=True, _print=False, _file=None, action_list=[]):

        self.action = action
        self.visited = np.zeros((exp_params.n_states))
        self.states = dict()
        self.n = 0
        self._value = 0
        self._total = 0
        self._rollout = rollout
        self._print = False
        self._open = True if _file is None else False
        self.file = _file

    def value(self, uct=False):
        # if uct and self.n == 0:
        #     return float('+inf') 
        if self.n != 0:
            return self._total / self.n
        return 0

    def average_value(self, uct=False):
        val = 0
        for s in self.states:
            val += exp_params.gamma * (self.states[s].value_() * self.visited[s])
        avg_val = 0
        if self.n != 0:
            avg_val += val / self.n
        val_old = 0
        for s in self.states:
            val_old += exp_params.gamma * (self.states[s].value_old() * self.visited[s])
        avg_val_old = 0
        if self.n != 0:
            avg_val_old += val_old / self.n
        return (self.value(), avg_val, avg_val_old)

    def simulate(self, state, depth, k):

        """Simulates from the action (ie state-action node)"""
        self.n += 1

        # Check if the transition is null (possibly by using the MLE) or if the depth has been reached
        sum_p = np.sum(exp_params.P[state][self.action])
        if (sum_p <= 0 + 0.00001) or depth > exp_params.max_depth:
            # reward = exp_params.R[state][self.action]
            reward = exp_params.R[state][self.action]

            # In this case just return the reward
            self._value += k * reward
            return k * reward

        s_prime = np.random.choice(range(0, exp_params.n_states),
                                   p=exp_params.P[state][self.action])

        # _rollout is a boolean: if it is true, we use a rollout strategy, otherwise we just expand all states
        # and keep simulating (very slow and hard on memory)
        if self._rollout:
            # Has not been visited before: rollout
            if self.visited[s_prime] == 0:
                node = StateNodeNew(state=s_prime)
                self.states[s_prime] = node
                delayed_reward = exp_params.gamma * self.states[s_prime].rollout(s_prime, depth + 1)

            else:
                delayed_reward = exp_params.gamma * self.states[s_prime].simulate(depth + 1)

        else:
            # expand if not visited
            if self.visited[s_prime] == 0:
                node = StateNodeNew(state=s_prime, _file=self.file)
                self.states[s_prime] = node

        self.visited[s_prime] += 1

        # again, if we do not rollout we need to simulate again to get the delayed reward
        if not self._rollout:
            delayed_reward = exp_params.gamma * self.states[s_prime].simulate(depth + 1)

        actual_reward = exp_params.R[state][self.action]

        reward = (actual_reward + delayed_reward)

        self._total += k * reward

        return k * reward


class StateNode():
    def __init__(self, state=0, initial_n=0, initial_val=0, root=False, _file=None):
        self.root = root
        self.file = _file
        self.state = state
        # Calculate safe actions
        self.safe_actions = np.where(exp_params.mask[state])
        self.non_safe_actions = np.where(~exp_params.mask[state])
        self.to_simulate = []  # for StateNodeNew
        self.p_nonsafe = np.sum(exp_params.pi_b_masked[state])
        self.p_safe = 1 - self.p_nonsafe
        self.p_pick_bootstrapped = 0
        self.initial_n = initial_n
        self.initial_val = initial_val
        self.fast_ucb_limit = exp_params.fast_ucb_limit
        self.totalN = 0  # for uct
        self._oldtotal = 0
        # I change here UCT
        self.n = 0
        self.children = np.zeros((exp_params.n_actions, exp_params.n_states))
        self.children_instances = dict()
        self.actions = dict()
        self.create_children_all(self.state)
        self.ns = [self.initial_n for i in range(0, exp_params.n_actions)]
        self.values = [self.initial_val for i in range(0, exp_params.n_actions)]
        self._value = 0  # Total value
        self.type_node = exp_params.state_node_type
        self.bootstrapped_values = [0 for i in range(0, exp_params.n_actions)]
        self.non_bootstrapped_value = 0
        self.rewards = []

    # Q must contain the tree Q values
    def new_pi_state_node(self, q):
        """Build the new pi from the Q-values"""
        pi = exp_params.pi_b_masked[self.state].copy()
        pi_b_masked_sum = np.sum(exp_params.pi_b_masked[self.state])

        # pi_b is not completely non safe for this state
        if pi_b_masked_sum < 1 - 0.0001:
            safe_indices = np.where(exp_params.mask[self.state])[0]
            pi[safe_indices[np.argmax(q[safe_indices])]] = 1 - pi_b_masked_sum

        return pi

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
    def stats(self):
        """Return the states: q-values, number of visits, value, old value (only UCT, no our formula)"""
        return [a.value() for a in self.actions.values()], self.ns, self.value_(), self.value_old()

    def create_children(self, action):
        """Expand"""
        newNode = ActionNode(action)
        self.actions[action] = newNode

    def create_children_all(self, state):
        """Expand all actions"""
        for a in range(0, exp_params.n_actions):
            if self.root:
                #file = open(f'Action={a}', 'a')
                newNode = ActionNode(a, state, _print=False)
            else:
                newNode = ActionNode(a, state, _print=False)
            self.actions[a] = newNode

    def simulate(self, depth):
        """Simulate"""

        action = self.pick_action(self.p_nonsafe)
        # reward = 0
        # Boolean: is the action bootstrapped or not
        bootstrapped = action in self.non_safe_actions[0]

        # Constant to append to the value (see formula - either pi_b(s) or p)
        k = exp_params.ks[self.state][action]

        # Simulate from the action node
        reward = self.actions[action].simulate(self.state, depth=depth, k=1)  # 1 could be k

        # Update statistics (not trivial)
        self.update_node_stats(action, reward, k, bootstrapped=bootstrapped)  # Also update state node

        return 1 * reward  # 1 could be k

    def rollout(self, state, depth):

        act = np.sum(exp_params.pi_b_masked[state])

        # TODO change to drawing a random theta and then in case take bootstrapped or non bootstrapped
        action = self.pick_action(act, state=state)

        k = 1
        sum_p = np.sum(exp_params.P[state][action])
        if (sum_p <= 0 + 0.00001) or depth > exp_params.max_depth:
            reward = k * exp_params.R[state][action]

            return reward

        new_state = np.random.choice(range(0, exp_params.n_states), p=exp_params.P[state][action])

        actual_reward = exp_params.R[state][action]

        # RECURSIVE CALL
        delayed_reward = exp_params.gamma * self.rollout(new_state, depth + 1)

        tot = k * (actual_reward + delayed_reward)

        return tot

    def iterative_rollout(self, state, depth):
        gamma = 1
        reward = 0
        k = 1
        while depth <= exp_params.max_depth:

            # TODO change by using the baseline
            action = np.random.choice(range(0, exp_params.n_actions))

            if state in exp_params.terminals:
                reward += k * gamma * exp_params.R[state][action]

                break

            c = 0  # failsafe
            while exp_params.legal_actions[state][action] != 1 and c < 50:
                action = np.random.choice(range(0, exp_params.n_actions))
                c += 1
            if c >= 10:
                break

            reward += k * gamma * exp_params.R[state][action]

            gamma *= exp_params.gamma

            state = np.random.choice(range(0, exp_params.n_states), p=exp_params.P[state][action])
            depth += 1

        return reward

    def pick_nonbootstrapped_action(self, state, safe):  # Never picks a non safe action

        # selection with prior knowledge (baseline) if any n = 0
        if 0 in self.ns:
            indices = [i for i, x in enumerate(self.ns) if x == 0]
            probs = [1 / len(indices)] * len(indices)
            selected_action = random.choices(population=indices, weights=probs)[0]

            return selected_action

        # Must compute it
        else:

            updated_vals = [
                self.actions[a].value(uct=True) + exp_params.exp_cost * np.sqrt(np.log(self.totalN) / self.ns[a]) for a
                in self.safe_actions[0]]

        return self.safe_actions[0][np.argmax(updated_vals)]

    def pick_action(self, custom_prob, state=None):

        in_rollout = True
        if state is None:
            in_rollout = False
            state = self.state
            nonsafe = self.non_safe_actions
            safe = self.safe_actions
        else:

            nonsafe = np.where(~exp_params.mask[state])
            safe = np.where(exp_params.mask[state])

        unsafe_to_pick = [a for a in range(0, exp_params.n_actions) if a in nonsafe[0]]  # legal
        prob_baseline = np.take(exp_params.pi_b_masked[state].copy(), unsafe_to_pick)


        """Action selection strategy"""
        # Custom prob can be something different: we use, in general, the probability of all bootstrapped actions (1-p)
        safe_unsafe = np.random.random()
        if len(nonsafe[0]) != 0 and safe_unsafe <= custom_prob:
            # Possible non_safe actions to pick
            to_pick = [a for a in range(0, exp_params.n_actions) if a in nonsafe[0]]  # legal

            if len(to_pick) != 0:
                probs = 0
                probs = np.take(exp_params.pi_b_masked[state].copy(), to_pick)
                probs = probs / (np.sum(probs))

                # Now the probability is 1 over all non safe actions
                r = np.random.choice(to_pick, p=probs)

                return r
        # Else, we pick from the safe actions, by using UCT
        if len(safe[0]) != 0:
            if not in_rollout:

                nonbt = self.pick_nonbootstrapped_action(state, safe)

                return nonbt
            else:

                rc = np.random.choice(safe[0])

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
        for a in self.non_safe_actions[0]:
            if self.ns[a] != 0:
                bval += self.bootstrapped_values[a] / self.ns[a]
        nonbval = 0
        if self.totalN != 0:
            nonbval += self.non_bootstrapped_value / self.totalN

        # act_values = [self.actions[a].value for a in self.actions]

        # return np.sum(self.bootstrapped_values) + self.non_bootstrapped_value
        return bval + nonbval

    def value_comparison(self):
        """Debug function to check if the stored value is the correct one"""
        r2 = np.dot(exp_params.pi_b_masked[self.state],
                    [a.value() for a in self.actions.values()]) + self.p_safe * self.non_bootstrapped_value
        return self.value_(), r2

    def value_old(self):
        """Value as computed by the classic UCT strategy, not ours"""
        if self.n != 0:
            return self._oldtotal / self.n
        return 0


class Mcts:
    """
    Class for MCTS-SPIBB. Its purpose is to hold the root of the trees, which will 'start' from an instance of StateNodeNew
    """

    def __init__(self, gamma, nb_states, nb_actions, baseline, MLE_T, R, mask, q_values_baseline, budget, seed,
                 q_pib_est=None, errors=None, epsilon=None, max_nb_it=99999, n_sims=4096,
                 max_depth=15, exploration_costant=10, type_node='old', states_to_sim=[],
                 final_states=[]):

        self.final_states = final_states
        self.states_to_sim = states_to_sim
        self.nodes = dict()
        # self.q_values = np.zeros((nb_states, nb_actions))
        self.q_values = q_values_baseline
        self.touched = np.zeros((nb_states,))
        self.v = np.zeros((nb_states,))
        self.v_old = np.zeros((nb_states,))
        self.ns = np.zeros((nb_states, nb_actions))
        self.max_depth = max_depth
        self.gamma = gamma
        self.n_sims = n_sims
        self.seed = seed
        self.nb_actions = nb_actions
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.pi = baseline
        self.P = MLE_T
        self.R = R
        self.pi_b_masked = self.pi.copy()
        self.pi_b_masked[mask] = 0
        # True <=> Safe
        self.mask = mask
        self.budget = budget

        self.exploration_costant = exploration_costant
        self.errors = errors
        self.epsilon = epsilon
        self.max_nb_it = max_nb_it
        self.type_node = type_node

    def computeTree(self, s):
        """
        Parameters: 
            s The state for which we want to compute the tree
        """
        # Global instance to hold the experiment parameters (save memory)
        global exp_params
        exp_params = ExpParams(self.nb_actions, self.nb_states, self.exploration_costant, self.gamma,
                               self.max_depth, self.P, self.R, self.final_states, self.mask, self.pi_b_masked, self.n_sims,
                               self.type_node)

        # Root node
        root = StateNodeNew(state=s, root=True)
        # Return all info for the state (n_visits, total_return, value etc.) building the tree from the root
        # return root.build_tree(self.n_sims)
        return root.build_tree(self.budget)

    def fit(self, state_to_sim=None, cores=0):
        """
        Compute the policy for all states
        """

        if state_to_sim is not None:
            state_sims = state_to_sim
        else:
            state_sims = self.states_to_sim

        # Hold the results for each state
        results = dict()

        # Multi-core implementation
        if cores > 0:
            results_ = []
            p = multiprocessing.Pool(processes=cores)
            results_ = p.map(self.computeTree, state_sims)
            p.close()
            p.join()
            c = 0
            for s in state_sims:
                results[s] = results_[c]
                c += 1
        # Single core
        else:
            for s in state_sims:
                print("state: %s" % s)
                results[s] = self.computeTree(s)
        # Now results hold various statistics., We unfold them
        for s in state_sims:
            self.q_values[s] = results[s][0]
            self.ns[s] = results[s][1]
            self.v[s] = results[s][2]
            self.v_old[s] = results[s][3]
            self.pi[s] = self.new_pi(s, np.array(results[s][0]))

    # Q must contain the tree Q values
    def new_pi(self, s, q):
        """Build the new pi from the Q-values"""
        #        print("new_pi()")

        pi = self.pi_b_masked[s].copy()

        pi_b_masked_sum = np.sum(self.pi_b_masked[s])
        # pi_b is not completely non safe for this state
        if pi_b_masked_sum < 1 - 0.0001:
            #            print("new_pi() - if")
            safe_indices = np.where(self.mask[s])[0]
            pi[safe_indices[np.argmax(q[safe_indices])]] = 1 - pi_b_masked_sum

        return pi

    @staticmethod
    def create_node_index(prev_index, state, action):
        if prev_index == '':
            comma_or_not = ''
        else:
            comma_or_not = ','
        if state == None:
            return f'{prev_index}{comma_or_not}{action}'
        return f'{prev_index}{comma_or_not}{state},{action}'
