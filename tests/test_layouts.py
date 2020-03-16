from  grandalf.graphs  import graph_core,Vertex,Edge,Graph
from  grandalf.layouts import SugiyamaLayout,DigcoLayout,VertexViewer,Layer,DummyVertex
from grandalf.routing import EdgeViewer, route_with_rounded_corners

def test_001_Sugiyama(sample_G02):
    gr = graph_core(*sample_G02)
    for v in gr.V(): v.view = VertexViewer(10,10)
    sug = SugiyamaLayout(gr)
    sug.init_all(roots=[gr.sV[0]],inverted_edges=[])
    for _ in sug.draw_step():
        for v,x in sug.grx.items():
            print (x, v.view.xy)

def test_002_Digco(sample_G02):
    gr  = graph_core(*sample_G02)
    for v in gr.V(): v.view = VertexViewer(10,10)
    dig  = DigcoLayout(gr)
    dig.init_all()
    dig.draw()
    for v in gr.sV: print (v,v.view.xy)

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
    for i in range(6):
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

    # not needed anymore...
    #r = filter(lambda x: len(x.e_in()) == 0, gr.sV)
    #if len(r) == 0:
    #    r = [gr.sV[0]]
    return gr, data_to_vertex

class CustomRankingSugiyamaLayout(SugiyamaLayout):

    def init_ranking(self,initial_ranking):
        '''
        :param dict{vertex:int} initial_ranking:
            The initial ranking of each vertex if passed
        '''
        self.initial_ranking = initial_ranking
        assert 0 in initial_ranking
        nblayers = max(initial_ranking.keys())+1
        self.layers = [Layer([]) for l in range(nblayers)]

    def _rank_init(self,unranked):
        assert self.dag

        if not hasattr(self, 'initial_ranking'):
            super()._rank_init(unranked)
        else:
            for rank, vertices in sorted(self.initial_ranking.items()):
                for v in vertices:
                    self.grx[v].rank=rank
                    # add it to its layer:
                    self.layers[rank].append(v)

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
    gr, data_to_vertex = create_scenario()
    sug = SugiyamaLayout(gr)
    sug.route_edge = route_with_rounded_corners
    sug.init_all()
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
    gr, data_to_vertex = create_scenario()
    sug = CustomRankingSugiyamaLayout(gr)
    sug.route_edge = route_with_rounded_corners
    rank_to_data = {
        0: [data_to_vertex['v4'], data_to_vertex['v0']],
        1: [data_to_vertex['v1']],
        2: [data_to_vertex['v5']],
        3: [data_to_vertex['v2']],
        4: [data_to_vertex['v3']],
    }
    sug.init_ranking(rank_to_data)
    sug.init_all()
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
    gr, data_to_vertex = create_scenario()
    sug = CustomRankingSugiyamaLayout(gr)
    sug.route_edge = route_with_rounded_corners
    rank_to_data = {
        0: [data_to_vertex['v4'], data_to_vertex['v0']],
        1: [data_to_vertex['v5'], data_to_vertex['v1']],
        2: [data_to_vertex['v2']],
        3: [data_to_vertex['v3']],
    }
    try:
        sug.init_ranking(rank_to_data)
        sug.init_all()
    except ValueError as e:
        assert e.message == 'bad ranking'
