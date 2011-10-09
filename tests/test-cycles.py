#!/usr/bin/env python

from grandalf import *

g = utils.Dot().read('tests/samples/cycles.dot')[0]
V = {}
for k,v in g.nodes.iteritems():
    V[k]=graphs.Vertex(k)
    V[k].view = layouts.VertexViewer(10,10)
E = []
for e in g.edges:
    E.append(graphs.Edge(V[e.n1.name],V[e.n2.name]))

G = graphs.Graph(V.values(),E)

sg = layouts.SugiyamaLayout(G.C[0])
gr = sg.g

r = filter(lambda x: len(x.e_in())==0, gr.sV)
L = gr.get_scs_with_feedback(r)

print 'roots',[x.data for x in r]
for s in L:
  print [x.data for x in s]
