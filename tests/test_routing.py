import pytest

from  grandalf.graphs  import *
from  grandalf.layouts import *
from  grandalf.routing import *

def test_splines(capsys,sample_G02):
    gr  = graph_core(*sample_G02)
    for  v in gr.V(): v.view = VertexViewer(10,10)
    sug  = SugiyamaLayout(gr)
    sug.init_all(roots=[gr.sV[0]],inverted_edges=[])
    i=0
    for  s in sug.draw_step():
        print ('--- step %d '%i +'-'*20)
        for v,x in sug.grx.items():
            print (x, v.view.xy)
        i+=1
    for  e in gr.E():
        e.view = EdgeViewer()
    sug.route_edge  = route_with_splines
    sug.draw_edges()
    for  e in sug.g.E():
        print ('edge (%s -> %s) :'%e.v)
        print (e.view._pts)
        print (e.view.splines)

    dig  = DigcoLayout(gr)
    dig.init_all()
    dig.draw()
    for  v in gr.sV: print (v,v.view.xy)

def test_rounded_corners(capsys,sample_G02):
    gr  = graph_core(*sample_G02)
    for  v in gr.V(): v.view = VertexViewer(10,10)
    sug  = SugiyamaLayout(gr)
    sug.init_all(roots=[gr.sV[0]],inverted_edges=[])
    i=0
    for  s in sug.draw_step():
        print ('--- step %d '%i +'-'*20)
        for v,x in sug.grx.items():
            print (x, v.view.xy)
        i+=1
    for  e in gr.E():
        e.view = EdgeViewer()
    sug.route_edge  = route_with_rounded_corners
    sug.draw_edges()
    for  e in sug.g.E():
        print ('edge (%s -> %s) :'%e.v)
        print (e.view._pts)

    dig  = DigcoLayout(gr)
    dig.init_all()
    dig.draw()
    for  v in gr.sV: print (v,v.view.xy)
