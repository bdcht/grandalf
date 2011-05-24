#!/usr/bin/env python

from grandalf.graphs import Vertex,Edge

# define very simple graphs :

v = map(chr,range(ord('a'),ord('c')+1))
V = [Vertex(x) for x in v]
D = dict(zip(v,V))
e = ['ab','ac']
E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
G01 = (V,E)

V = [Vertex(x) for x in v]
D = dict(zip(v,V))
e = ['ab','ac','bc']
E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
G02 = (V,E)

v.append('d')
V = [Vertex(x) for x in v]
D = dict(zip(v,V))
e = ['ab','bc','cd','bd','da']
E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
G03 = (V,E)

v = map(chr,range(ord('a'),ord('z')+1))
V = [Vertex(x) for x in v]
D = dict(zip(v,V))
e = ['ab','bc','cd','de','ef','fg','gh','hi','ij','jk','kl','lm',
     'mn','no','op','pq','qr','rs','st','tu','uv','vw','wx','xy','yz','za']
E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
G04 = (V,E)
