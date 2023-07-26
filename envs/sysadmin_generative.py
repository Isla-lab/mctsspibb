import pickle
import numpy as np
from numpy import random


class SysAdmin:
    def __init__(self, n_machines):
        self.n_machines = n_machines
        self.n_states = 2 ** n_machines
        self.n_actions = n_machines + 1
        self.state = self.reset()

    def save_object(obj, filename):
        with open(filename, 'wb') as out:  # Overwrites any existing file.
            pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)

    def decode(self, i):
        out = bin(i)[2:].rjust(self.n_machines, '0')
        return map(int, reversed(out))

    def neighbor_of(self, m, n_machines):
        return [(m - 1) % n_machines, (m + 1) % n_machines]

    def encode(self, statuses):
        i = 0
        assert self.n_machines == len(statuses)
        for m in range(self.n_machines):
            if statuses[m]:
                i += 2 ** m
        return i

    def step(self, state, action):
        state = np.array(list(self.decode(state)))
        machines_on = np.where(state == 1)[0]
        prob_on = [m if m == 1 else 0 for m in state]
        machines_off = np.where(state == 0)[0]

        if action < self.n_machines:

            state[action] = 1
            prob_on[action] = 1
            machines_on = np.where(state == 1)[0]
            machines_off = np.where(state == 0)[0]

        next_state = state.copy()
        if machines_on.size != 0:
            fail_prob = 0.05
            for m in machines_on:
                if m != action:
                    for neighbor in self.neighbor_of(m, self.n_machines):
                        if neighbor in machines_off:
                            fail_prob += 0.3
                prob_on[m] = max(1 - fail_prob, 0)

            for s in range(len(state)):
                if prob_on[s] != 0 and random.random() < 1 - prob_on[s]:
                    prob_on[s] = 0
                    next_state[s] = 0

        self.state = next_state
        reward = self.reward_function(self.state, action)
        next_state = self.encode(self.state)

        return next_state, reward, False

    def reward_function(self, state, action):
        return np.sum(state) - (action < self.n_machines)
        # return sum(np.array(list(self.decode(state)))) - len(np.where(np.array(list(self.decode(state))) == 0)[0]) \
        #        - (action < self.n_machines)

    def reset(self):
        self.state = 0

        return self.state