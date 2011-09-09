#!/usr/bin/env python

import  pdb

from  grandalf.graphs  import *
from  grandalf.layouts import *
from  grandalf.utils   import IDA

mypdb  = pdb.Pdb()

def test_lexer():
    ida = IDA()
    ida.lexer.build(optimize=1,lextab=None)
    ida.lexer.test('''
    graph: {
      title: "ida format"
      colorentry 32: 0 0 0
      node: {title:"n1" label:"a"}
      node: {title:"n2" label:"b"}
      edge: {sourcename: "n1" targetname:"n2"
    }
    ''')

if __name__ == '__main__':

    ast  = IDA().read('samples/ida12487.tmp')

    print "testing graph %s :"%ast.name,
    V = {}
    E = []
    for k,x in ast.nodes.iteritems():
        try:
            v = Vertex(x.label)
        except (KeyError,AttributeError):
            v = Vertex(x.title)
        v.view = VertexViewer(10,10)
        V[x.title] = v
    print len(V)
    for e in ast.edges:
        v1 = V[e.sourcename]
        v2 = V[e.targetname]
        E.append(Edge(v1,v2))
    #mypdb.set_trace()
    G = Graph(V.values(),E)
    print "  [%d vertices]"%G.order()
    print "  [%d groups]"%len(G.C)
    for gr in G.C:
        sug = SugiyamaLayout(gr)
        sug.init_all()
        sug.draw()
        #for s in sug.draw_step(): pass
        for v,x in sug.grx.iteritems():
            label = v.data if hasattr(v,'data') else '*'
            print label, x, v.view.xy
        print 'Sugiyama drawing done.'.ljust(80,'_')

