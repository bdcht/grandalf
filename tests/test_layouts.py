#!/usr/bin/env python

from  grandalf.graphs  import *
from  grandalf.layouts import *
from grandalf.routing import EdgeViewer, route_with_rounded_corners
from  samples.G0 import G02


def test_layouts():
    gr  = graph_core(*G02)
    for  v in gr.V(): v.view = VertexViewer(10,10)
    sug  = SugiyamaLayout(gr)
    sug.init_all(roots=[gr.sV.o[0]],inverted_edges=[])
    i=0
    for  s in sug.draw_step():
        print '--- step %d '%i +'-'*20
        for v,x in sug.grx.iteritems():
            print x, v.view.xy
        i+=1

    dig  = DigcoLayout(gr)
    dig.init_all()
    dig.draw()
    for  v in gr.sV: print v,v.view.xy


def create_scenario():
    '''
    Create something as:
    
    v4      v0
     \     / |
      \   v1 |
       \ /   |
       v5   /
        \  /
        v2
        |
        v3
    '''
    E = []
    data_to_vertex = {}

    vertices = []
    for i in xrange(6):
        data = 'v%s' % (i,)
        v = Vertex(data)
        data_to_vertex[data] = v
        v.view = VertexViewer(100, 50)
        vertices.append(v)

    edge = Edge(vertices[0], vertices[1])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[0], vertices[2])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[1], vertices[5])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[2], vertices[3])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[4], vertices[5])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[5], vertices[2])
    edge.view = EdgeViewer()
    E.append(edge)

    G = Graph(vertices, E)
    assert len(G.C) == 1
    gr = G.C[0]
    
    r = filter(lambda x: len(x.e_in()) == 0, gr.sV)
    if len(r) == 0:
        r = [gr.sV.o[0]]
    return gr, r, data_to_vertex

class CustomRankingSugiyamaLayout(SugiyamaLayout):
    

    def init_all(self, roots=None, inverted_edges=None, cons=False, initial_ranking=None):
        '''
        :param dict{vertex:int} initial_ranking:
            The initial ranking of each vertex if passed
        '''
        if initial_ranking is not None:
            self.initial_ranking = initial_ranking
            
        SugiyamaLayout.init_all(self, roots=roots, inverted_edges=inverted_edges, cons=cons)
        
    def _rank_init(self,unranked):
        assert self.dag
        
        if not hasattr(self, 'initial_ranking'):
            SugiyamaLayout._rank_init(self, unranked)
        else:
            for rank, vertices in sorted(self.initial_ranking.iteritems()):
                for v in vertices:
                    self.grx[v].rank=rank
                    # add it to its layer:
                    try:
                        self.layers[rank].append(v)
                    except IndexError:
                        self.layers.append(Layer([v]))

def _compute_rank_to_data(sug):
    rank_to_data = {}
    for rank, layer in enumerate(sug.layers):
        data = rank_to_data[rank] = []
        for v in layer:
            if isinstance(v, DummyVertex):
                continue
            data.append(v.data)
    return rank_to_data
    

def test_sugiyama_ranking():
    gr, r, data_to_vertex = create_scenario()
    
    sug = SugiyamaLayout(gr)
    sug.route_edge = route_with_rounded_corners
    sug.init_all(roots=r, inverted_edges=filter(lambda x: x.feedback, gr.sE))
            
    # rank 0: v4      v0
    #          \     / |
    # rank 1:   \   v1 |
    #            \ /   |
    # rank 2:    v5   /
    #             \  /
    # rank 3:     v2
    #             |
    # rank 4:     v3
    rank_to_data = _compute_rank_to_data(sug)
    assert rank_to_data == {
        0: ['v4', 'v0'], 
        1: ['v1'], 
        2: ['v5'], 
        3: ['v2'], 
        4: ['v3'], 
    }
    sug.draw(N=10)
    
def test_sugiyama_custom_ranking():
    gr, r, data_to_vertex = create_scenario()
    
    sug = CustomRankingSugiyamaLayout(gr)
    sug.route_edge = route_with_rounded_corners
    
    rank_to_data = {
        0: [data_to_vertex['v4'], data_to_vertex['v0']], 
        1: [data_to_vertex['v1']], 
        2: [data_to_vertex['v5']], 
        3: [data_to_vertex['v2']], 
        4: [data_to_vertex['v3']], 
    }
    sug.init_all(roots=r, inverted_edges=filter(lambda x: x.feedback, gr.sE), initial_ranking=rank_to_data)
            
    # rank 0: v4      v0
    #          \     / |
    # rank 1:   \   v1 |
    #            \ /   |
    # rank 2:    v5   /
    #             \  /
    # rank 3:     v2
    #             |
    # rank 4:     v3
    rank_to_data = _compute_rank_to_data(sug)
    assert rank_to_data == {
        0: ['v4', 'v0'], 
        1: ['v1'], 
        2: ['v5'], 
        3: ['v2'], 
        4: ['v3'], 
    }
    sug.draw(N=10)
    
def test_sugiyama_custom_ranking2():
    gr, r, data_to_vertex = create_scenario()
    
    sug = CustomRankingSugiyamaLayout(gr)
    sug.route_edge = route_with_rounded_corners
    
    rank_to_data = {
        0: [data_to_vertex['v4'], data_to_vertex['v0']], 
        1: [data_to_vertex['v5'], data_to_vertex['v1']], 
        2: [data_to_vertex['v2']], 
        3: [data_to_vertex['v3']], 
    }
    sug.init_all(roots=r, inverted_edges=filter(lambda x: x.feedback, gr.sE), initial_ranking=rank_to_data)
            
    # rank 0: v4        v0
    #          \        | \
    # rank 1:   v5 --  v1 |
    #            \   /   /
    # rank 2:       v2
    #               |
    # rank 3:       v3
    rank_to_data = _compute_rank_to_data(sug)
    assert rank_to_data == {
        0: ['v4', 'v0'], 
        1: ['v5', 'v1'], 
        2: ['v2'], 
        3: ['v3'], 
    }
    sug.draw(N=10)
