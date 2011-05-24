#!/usr/bin/env python
from grandalf.graphs import Vertex,Edge

# horizontal coord assigment verifier:
v = range(1,24)
V = [Vertex(x) for x in map(str,v)]
D = dict(zip(v,V))
e = [(1,13), (1,21), (1,4), (1,3), (2,3), (2,20),
     (3,4), (3,5), (3,23),
     (4,6), (5,7),
     (6,8), (6,16), (6,23), (7,9),
     (8,10), (8,11), (9,12),
     (10,13), (10,14), (10,15), (11,15), (11,16), (12,20),
     (13,17), (14,17), (14,18), (16,18), (16,19), (16,20),
     (18,21), (19,22),
     (21,23), (22,23)
    ]
E = [Edge(D[x],D[y]) for x,y in e]
G3= (V,E)
