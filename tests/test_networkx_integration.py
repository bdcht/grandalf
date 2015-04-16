from grandalf.graphs import Vertex, Edge, Graph
from grandalf.utils.nx import *


def test_networkx():
    try:
        import networkx
    except ImportError:
        return  # Only run test if networkx is present
    
    v = ('a', 'b', 'c', 'd')
    V = [Vertex(x) for x in v]
    D = dict(zip(v, V))
    e = ['ab', 'ac', 'bc', 'cd']
    E = [Edge(D[xy[0]], D[xy[1]], data=xy) for xy in e]
    
    g = Graph(V, E)
    
    nx_graph = convert_grandalf_graph_to_networkx_graph(g)

    assert set(n for n in nx_graph.nodes()) == set(v)  # nodes returns the nodes available (node data)
    assert set(n[0] + n[1] for n in nx_graph.edges()) == set(e)  # edges returns list(tuple(node1, node2))

    assert nx_graph.number_of_edges() == len(E)
    assert nx_graph.number_of_nodes() == len(V)
    
    # Now, let's go back from networkx to grandalf
    grandalf_graph = convert_nextworkx_graph_to_grandalf(nx_graph)
    
    assert len(list(grandalf_graph.V())) == len(v)
    assert len(list(grandalf_graph.E())) == len(E)
    assert set(n.data for n in grandalf_graph.V()) == set(v)
    assert set(n.v[0].data + n.v[1].data for n in grandalf_graph.E()) == set(e)
    
