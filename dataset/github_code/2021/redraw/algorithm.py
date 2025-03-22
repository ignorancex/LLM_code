import util
import numpy as np
import itertools
from scipy import spatial

DELTA = 0.001
K = 1000
EPSILON = 0.0025
C_VERT = 1
C_HOR = 5
C_PAR = 0.005
C_ANG = 0.05
C_DIST = 1


def round(G, show=False, parallelize=True):
    print("-> Node Step")
    for x in range(K):
        total_stress=0
        # Vertical forces
        for x in G.nodes():
            for y in G.nodes():
                if x != y and G.has_edge(x, y):
                    posx = G.pos(x)
                    posy = G.pos(y)
                    vert_dist = posx[0] - posy[0]
                    hor_dist = np.linalg.norm(posx[1:] - posy[1:])
                    spring = - C_VERT * ((1 + hor_dist) / vert_dist - 1)
                    f = spring * DELTA
                    total_stress += 2*abs(f)
                    G.add_v_force(x, - f)
                    G.add_v_force(y, f)

        # Attracting forces of chains
        for x in G.nodes():
            for y in G.nodes():
                if x != y and G.is_comp(x, y):
                    posx = G.pos(x)[1:]
                    posy = G.pos(y)[1:]
                    dist = np.linalg.norm(posx - posy)
                    factor = min(dist ** 2, C_HOR)*DELTA
                    direct = (posy - posx)
                    total_stress += 2*sum(abs(direct*factor))
                    G.add_h_force(x, + direct * factor)
                    G.add_h_force(y, - direct * factor)

        # Repelling forces between incomparable elements
        for x, y in itertools.combinations(G.nodes(), 2):
            if not G.is_comp(x, y) and not G.is_comp(y, x):
                posx = G.pos(x)
                posy = G.pos(y)
                vert_dist = posx[0] - posy[0]
                hor_dist = np.linalg.norm(posy[1:] - posx[1:])
                if hor_dist != 0:
                    direct = (posy[1:] - posx[1:])/hor_dist
                    spring = - (C_HOR / (hor_dist))*DELTA
                    total_stress += 2*sum(abs(spring*direct))
                    G.add_h_force(x, + spring * direct)
                    G.add_h_force(y, - spring * direct)

        total = G.apply_forces()
        if(total < EPSILON):
            break

    G.correct_offset()

    if parallelize:
        print("-> Line Step")
        for x in range(K):
            for x in G.edges():
                for y in G.edges():
                    if x != y and x[0] != x[1] and y[0] != y[1]:
                        v1 = G.pos(x[0]) - G.pos(x[1])
                        v2 = G.pos(y[0]) - G.pos(y[1])
                        if np.all(v1 == 0) or np.all(v2 == 0):
                            continue
                        sim = spatial.distance.cosine(v1, v2)
                        if sim < C_PAR:
                            a0 = G.pos(x[1]) - G.pos(x[0])
                            b0 = G.pos(y[1]) - G.pos(y[0])
                            a1 = a0[1:]/a0[0]
                            b1 = b0[1:]/b0[0]
                            v = (1-sim*1/C_PAR) * (a1 - b1)
                            G.add_h_force(x[0], - v * DELTA)
                            G.add_h_force(x[1], + v * DELTA)
                            G.add_h_force(y[0], + v * DELTA)
                            G.add_h_force(y[1], - v * DELTA)

            for x in G.edges():
                for y in G.edges():
                    if x != y and x[0] != x[1] and y[0] != y[1]:
                        v1 = G.pos(x[0]) - G.pos(x[1])
                        v2 = G.pos(y[0]) - G.pos(y[1])
                        if x[0] == y[0]:
                            sim = spatial.distance.cosine(v1, v2)
                            if 0 < sim < C_ANG:
                                a0 = G.pos(x[1]) - G.pos(x[0])
                                b0 = G.pos(y[1]) - G.pos(y[0])
                                a1 = a0[1:]/a0[0]
                                b1 = b0[1:]/b0[0]
                                v = (1 - sim * 1/C_ANG) * (a1 - b1)
                                G.add_h_force(x[1], - v * DELTA)
                                G.add_h_force(y[1], + v * DELTA)
                        if x[1] == y[1]:
                            sim = spatial.distance.cosine(v1, v2)
                            if 0 < sim < C_ANG:
                                a0 = G.pos(x[1]) - G.pos(x[0])
                                b0 = G.pos(y[1]) - G.pos(y[0])
                                a1 = a0[1:]/a0[0]
                                b1 = b0[1:]/b0[0]
                                v = (1 - sim * 1/C_ANG) * (a1 - b1)
                                G.add_h_force(x[0], + v * DELTA)
                                G.add_h_force(y[0], - v * DELTA)

            for x in G.edges():
                for p in G.nodes():
                    if G.pos(x[0])[0] > G.pos(p)[0] > G.pos(x[1])[0]:
                        a = x[0]
                        b = x[1]
                        pa = G.pos(p) - G.pos(a)
                        ba = G.pos(b) - G.pos(a)
                        t = np.dot(pa, ba)/np.dot(ba, ba)
                        d = np.linalg.norm(pa - t * ba)
                        if(d < C_DIST):
                            G.add_force(p, C_DIST * ((pa - t * ba)/d*DELTA))
                            G.add_force(a, - C_DIST * ((pa - t * ba)/d*DELTA/2))
                            G.add_force(b, - C_DIST * ((pa - t * ba)/d*DELTA/2))

            total = G.apply_forces()
            if(total < EPSILON):
                break

    G.correct_offset()
    if show:
        G.show()


def compute_drawing(filename, dimension, show_result=False):
    G = util.Order(filename, dimension)

    for x in range(dimension, 2, -1):
        print("Dimension {:d}:".format(x))
        round(G)
        print("-> Reduction Step")
        G.dimension_stepdown()
    print("Dimension 2:")
    round(G, show_result)

    return G


def main():
    filename = "./data/mushroom15_8.cxt"
    compute_drawing(filename, 5, True)


if __name__ == '__main__':
    main()
