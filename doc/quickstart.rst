===============
Getting started
===============

A very simple example::

   In [1]: from grandalf.graphs import Vertex,Edge,Graph
   In [2]: V = [Vertex(data) for data in range(10)]
   In [3]: X = [(0,1),(0,2),(1,3),(2,3),(4,0),(1,4),(4,5),(5,6),(3,6),(3,7),(6,8),(7,8),(8,9),(5,9)]
   In [4]: E = [Edge(V[v],V[w]) for (v,w) in X]
   In [5]: g = Graph(V,E)
   In [6]: g.C
   Out[6]: [<grandalf.graphs.graph_core at 0x7f73bde4a470>]
   In [7]: print([v.data for v in g.path(V[1],V[9])])
   [1, 4, 5, 9]
   In [8]: g.add_edge(Edge(V[9],Vertex(10)))
   Out[8]: <grandalf.graphs.Edge at 0x7f73bde4aa58>
   In [9]: g.remove_edge(V[5].e_to(V[9]))
   Out[9]: <grandalf.graphs.Edge at 0x7f73bde4a7b8>
   In [10]: print([v.data for v in g.path(V[1],V[9])])
   [1, 3, 6, 8, 9]
   In [11]: g.remove_vertex(V[8])
   Out[11]: <grandalf.graphs.Vertex at 0x7f73bde4a208>
   In [12]: len(g.C)
   Out[12]: 2
   In [13]: print(g.path(V[1],V[9]))
   None
   In [14]: for e in g.C[1].E(): print("%s->%s"%(e.v[0].data,e.v[1].data))
   9->10
   In [15]: from grandalf.layouts import SugiyamaLayout
   In [16]: class defaultview(object):
      ....:     w,h = 10,10
      ....:     
   In [17]: for v in V: v.view = defaultview()
   In [18]: sug = SugiyamaLayout(g.C[0])
   In [19]: sug.init_all(roots=[V[0]],inverted_edges=[V[4].e_to(V[0])])
   In [20]: sug.draw()
   In [21]: for v in g.C[0].sV: print("%s: (%d,%d)"%(v.data,v.view.xy[0],v.view.xy[1]))
   0: (0,5)
   1: (-45,35)
   2: (30,35)
   3: (30,65)
   4: (-30,65)
   5: (-30,95)
   6: (0,125)
   7: (15,95)



