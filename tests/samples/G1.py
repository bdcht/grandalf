#!/usr/bin/env python

from grandalf.graphs import Vertex,Edge

# define a very simple graph :
v = map(chr,range(ord('a'),ord('h')+1))
V = [Vertex(x) for x in v]
D = dict(zip(v,V))
e = ['ab','ae','af','bc','cd','dh','eg','fg','gh']
E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
G1= (V,E)
