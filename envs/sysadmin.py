from gym.envs.toy_text import discrete
import numpy as np
import sys
from six import StringIO


class SysAdminEnv(discrete.DiscreteEnv):
    """
    The SysAdmin Problem
    """

    def __init__(self, size):
        self.nM = size
        nM = size  # might be an argument to make it easier to scale the problem
        nS = 2 ** nM
        nA = nM + 1
        isd = np.zeros(nS)
        isd[nS-1] = 1  # always start with all machines running
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for s in range(nS):
            machines_on = list(self.decode(s))
            for a in range(nA):
                prob_on = np.zeros(nM)
                for m in range(nM):
                    if a == m:
                        prob_on[m] = 1
                    elif not machines_on[m]:
                        prob_on[m] = 0
                    else:
                        fail_prob = 0.05
                        for neighbor in self.neighbor_of(m):
                            if not machines_on[neighbor]:
                                fail_prob += 0.3
                        prob_on[m] = max(1 - fail_prob, 0)
                total_p = 0
                for ns in range(nS):
                    reward = sum(machines_on) - (a < nM)
                    p = 1
                    next_machines_on = list(self.decode(ns))
                    for m, status in enumerate(next_machines_on):
                        if status:
                            p *= prob_on[m]
                        else:
                            p *= (1-prob_on[m])
                    if p > 0:
                        total_p += p
                        P[s][a].append((p, ns, reward, False))
                assert abs(1 - total_p) < 0.0000001

        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def neighbor_of(self, m):
        # TODO: possibly specify a more general way of defining the topology of the network
        return [(m-1) % self.nM, (m+1) % self.nM]

    def encode(self, *statuses):
        i = 0
        assert self.nM == len(statuses)
        for m in range(self.nM):
            if statuses[m]:
                i += 2 ** m
        return i

    def decode(self, i):
        out = bin(i)[2:].rjust(self.nM, '0')
        return map(int, reversed(out))

    def get_true_parents(self):
        parents = dict()
        for a in range(self.nM + 1):
            for m in range(self.nM):
                if a == m:
                    parents[(m, a)] = tuple()
                else:
                    parents[(m, a)] = tuple(sorted(self.neighbor_of(m) + [m]))
        return parents

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        machines_on = list(self.decode(self.s))

        def ul(x):
            return "O" if x else "X"

        out = "".join([ul(x) for x in machines_on])

        outfile.write(out+"\n")
        if self.lastaction is not None:
            action_line = "".join("^" if i == self.lastaction else " " for i in range(self.nM + 1))
        else:
            action_line = ""
        outfile.write(action_line + "\n")

        if mode != 'human':
            return outfile
