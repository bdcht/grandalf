import pytest

from grandalf.graphs import *

V = [Vertex("b%d"%n) for n in range(1,16)]

def _E(b):
    b1,b2 = b
    return Edge(V[b1-1],V[b2-1])

E = map(_E, [ (1,2), (2,3), (2,4), (3,5), (4,5), (1,5),
           (5,6), (6,7), (6,12), (7,8), (7,9), (8,9), (8,10), (9,10), (10,11),
           (12,13), (13,14), (14,13), (14,15), (15,6)])

def test_part_001():
    G = Graph(V,E)
    g = G.C[0]
    P = g.partition()
    assert len(P)==3
    assert sum([len(p) for p in P])==g.order()
