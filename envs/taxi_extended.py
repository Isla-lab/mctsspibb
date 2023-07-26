import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

EXTENDED_MAP = [
    "+-------------------+",
    "|R: : | : :G: : |C: |",
    "| : : | : : : : | : |",
    "| : : | : : | : | : |",
    "| : : |W: : | : | : |",
    "| : : : : : |M: : : |",
    "| : : : : : | : : : |",
    "| | : : | : : : | : |",
    "| | : : | : : : | : |",
    "|Y| : : | : : : | : |",
    "| | : : |B: : : | :P|",
    "+-------------------+",
]

class TaxiExtendedEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(EXTENDED_MAP, dtype='c')

        self.locs = locs = [(0, 0), (0, 5), (0, 8), (3, 4), (4, 6), (8, 0), (9, 4), (9, 9)]

        nS = 7200
        nR = 10
        nC = 10
        maxR = nR-1
        maxC = nC-1
        isd = np.zeros(nS)
        nA = 6
        passenger_in_tax = len(locs)
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for row in range(10):
            for col in range(10):
                for passidx in range(9):
                    for destidx in range(8):
                        state = self.encode(row, col, passidx, destidx)
                        if passidx < 9 and passidx != destidx:
                            isd[state] += 1
                        for a in range(nA):
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            reward = -1
                            done = False
                            taxiloc = (row, col)

                            if a==0:
                                newrow = min(row+1, maxR)
                            elif a==1:
                                newrow = max(row-1, 0)
                            if a==2 and self.desc[1+row,2*col+2]==b":":
                                newcol = min(col+1, maxC)
                            elif a==3 and self.desc[1+row,2*col]==b":":
                                newcol = max(col-1, 0)
                            elif a==4: # pickup
                                if (passidx < passenger_in_tax) and (taxiloc == locs[passidx]):
                                    newpassidx = passenger_in_tax
                                else:
                                    reward = -10
                            elif a==5: # dropoff
                                if (taxiloc == locs[destidx]) and passidx==passenger_in_tax:
                                    done = True
                                    newpassidx = locs.index(taxiloc)
                                    reward = 20
                                elif (taxiloc in locs) and passidx==passenger_in_tax:
                                    newpassidx = locs.index(taxiloc)
                                else:
                                    reward = -10
                            newstate = self.encode(newrow, newcol, newpassidx, destidx)
                            P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, taxirow, taxicol, passloc, destidx):
        # (10) 10, 9, 8
        i = taxirow
        i *= 10
        i += taxicol
        i *= 9
        i += passloc
        i *= 8
        i += destidx
        return i

    def decode(self, i):
        out = []
        out.append(i % 8)
        i = i // 8
        out.append(i % 9)
        i = i // 9
        out.append(i % 10)
        i = i // 10
        out.append(i)
        assert 0 <= i < 10
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        if passidx < 4:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            pi, pj = self.locs[passidx]
            out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
        else: # passenger in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)

        di, dj = self.locs[destidx]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
