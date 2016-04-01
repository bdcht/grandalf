import pytest

from  grandalf import *

@pytest.mark.skipif(not utils.dot._has_ply,reason="requires ply module")
def test_001_lexer(capsys):
    d = utils.Dot()
    d.lexer.build()
    d.lexer.test('''
    strict diGRAPH "test" {
      NODE [color="red"]
      a b "c" ;
      a -> b [label="e1"];
      a -> "c";
      // subgraph for node b:
      subgraph b {
        d -> e [label=x, color="blue" type=spline] -> f [type="polylines"];
      }
    }
    grAPh dg {
      "x" - "y" - "z"
      /* edge attributes
         for all digraph dg */
      Edge [type=6 zone="x"];
    }
    ''')
    out, err = capsys.readouterr()
    lines = out.split()
    assert len(lines)==78
    assert all([x.startswith('LexToken') for x in lines])

@pytest.mark.skipif(not utils.dot._has_ply,reason="requires ply module")
def test_002_parser(sample_dot):
    print (utils.Dot().read(sample_dot))

@pytest.mark.skipif(not utils.dot._has_ply,reason="requires ply module")
def test_003_dg10(sample_dg10):
    L  = utils.Dot().read(sample_dg10)
    assert len(L)==10
    dglen = [49,51,52,52,49,51,78,20,76,9]
    for i in range(10):
        assert L[i].name=='dg_%d'%i
        assert len(L[i].nodes)==dglen[i]
    G  = []
    for  ast in L:
        V = {}
        E = []
        for k,x in ast.nodes.items():
            try:
                v = graphs.Vertex(x.attr['label'])
            except (KeyError,AttributeError):
                v = graphs.Vertex(x.name)
            v.view = layouts.VertexViewer(10,10)
            V[x.name] = v
        edgelist = []
        for e in ast.edges: edgelist.append(e)
        for edot in edgelist:
            v1 = V[edot.n1.name]
            v2 = V[edot.n2.name]
            E.append(graphs.Edge(v1,v2))
        G.append(graphs.Graph(V.values(),E))
        for gr in G[-1].C:
            sug = layouts.SugiyamaLayout(gr)
            sug.init_all()
            sug.draw()
