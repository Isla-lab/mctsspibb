from gym.envs.toy_text.taxi import TaxiEnv, MAP
import numpy as np
from gym.envs.toy_text import discrete


class TaxiDangerousEnv(TaxiEnv):
    """
    Adaptation of TaxiEnv where bumping on walls give a penalty

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        nS = 500
        nR = 5
        nC = 5
        maxR = nR - 1
        maxC = nC - 1
        isd = np.zeros(nS)
        nA = 6
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for row in range(5):
            for col in range(5):
                for passidx in range(5):
                    for destidx in range(4):
                        state = self.encode(row, col, passidx, destidx)
                        if passidx < 4 and passidx != destidx:
                            isd[state] += 1
                        for a in range(nA):
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            newrow2 = None
                            reward = -1
                            reward2 = -1
                            done = False
                            taxiloc = (row, col)

                            if a == 0:
                                newrow = min(row + 1, maxR)
                                if newrow == row:
                                    reward = -10
                                newrow2 = max(row - 1, 0)
                                if newrow2 == row:
                                    reward2 = -10
                            elif a == 1:
                                newrow = max(row - 1, 0)
                                if newrow == row:
                                    reward = -10

                                newrow2 = min(row + 1, maxR)
                                if newrow2 == row:
                                    reward2 = -10

                            elif a == 2:
                                if self.desc[1 + row, 2 * col + 2] == b":":
                                    newcol = min(col + 1, maxC)
                                if newcol == col:
                                    reward = -10
                            elif a == 3:
                                if self.desc[1 + row, 2 * col] == b":":
                                    newcol = max(col - 1, 0)
                                if newcol == col:
                                    reward = -10
                            elif a == 4:  # pickup
                                if passidx < 4 and taxiloc == locs[passidx]:
                                    newpassidx = 4
                                else:
                                    reward = -10
                            elif a == 5:  # dropoff
                                if (taxiloc == locs[destidx]) and passidx == 4:
                                    newpassidx = destidx
                                    done = True
                                    reward = 20
                                elif (taxiloc in locs) and passidx == 4:
                                    newpassidx = locs.index(taxiloc)
                                else:
                                    reward = -10
                            if newrow2 is None:
                                newstate = self.encode(newrow, newcol, newpassidx, destidx)
                                P[state][a].append((1.0, newstate, reward, done))
                            else:
                                newstate = self.encode(newrow, newcol, newpassidx, destidx)
                                P[state][a].append((0.8, newstate, reward, done))
                                newstate = self.encode(newrow2, newcol, newpassidx, destidx)
                                P[state][a].append((0.2, newstate, reward2, done))
                        if a in [0, 1]:
                            assert len(P[state][a]) == 2
                        else:
                            assert len(P[state][a]) == 1
                        assert abs(sum(p for p, _, _, _ in P[state][a]) - 1) < 0.00001

        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)
