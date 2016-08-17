import pytest
import os

samples_dir = os.path.join(os.path.dirname(__file__), 'samples')

def get_samples_file(filename):
    return os.path.join(samples_dir, filename)

samples_all = []

for R,D,F in os.walk(samples_dir):
    for f in F:
        filename = os.path.join(R,f)
        samples_all.append(filename)

@pytest.fixture(params=filter(lambda x:x[-4:]=='.dot',samples_all))
def sample_dot(request):
    return request.param

@pytest.fixture
def sample_cycle():
    return get_samples_file('cycles.dot')

@pytest.fixture
def sample_dg10():
    return get_samples_file('dg10.dot')

#------------------------------------------------------------------------------
# toys found in various papers

from grandalf.graphs import Vertex,Edge

@pytest.fixture
def sample_G01():
    v = 'abc'
    V = [Vertex(x) for x in v]
    D = dict(zip(v,V))
    e = ['ab','ac']
    E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
    return (V,E)

@pytest.fixture
def sample_G02():
    v = 'abc'
    V = [Vertex(x) for x in v]
    D = dict(zip(v,V))
    e = ['ab','ac','bc']
    E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
    return (V,E)

@pytest.fixture
def sample_G03():
    v = 'abcd'
    V = [Vertex(x) for x in v]
    D = dict(zip(v,V))
    e = ['ab','bc','cd','bd','da']
    E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
    return (V,E)

@pytest.fixture
def sample_G04():
    V = map(chr,range(ord('a'),ord('z')+1))
    D = dict(zip(v,V))
    e = ['ab','bc','cd','de','ef','fg','gh','hi','ij','jk','kl','lm',
         'mn','no','op','pq','qr','rs','st','tu','uv','vw','wx','xy','yz','za']
    E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
    return (V,E)

@pytest.fixture
def sample_G05():
    v = 'abcdefgh'
    V = [Vertex(x) for x in v]
    D = dict(zip(v,V))
    e = ['ab','ae','af','bc','cd','dh','eg','fg','gh']
    E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
    return (V,E)

@pytest.fixture
def sample_G06():
    v = range(30)
    V = [Vertex(x) for x in map(str,v)]
    D = dict(zip(v,V))
    e = [(0,5), (0,29), (0,6), (0,20), (0,4),
         (17,3), (5,2), (5,10), (5,14), (5,26), (5,4), (5,3),
         (2,23), (2,8), (14,10), (26,18), (3,4),
         (23,9), (23,24), (10,27), (18,13),
         (1,12), (24,28), (24,12), (24,15),
         (12,9), (12,6), (12,19),
         (6,9), (6,29), (6,25), (6,21), (6,13),
         (29,25), (21,22),
         (25,11), (22,9), (22,8),
         (11,9), (11,16), (8,20), (8,16), (15,16), (15,27),
         (16,27),
         (27,19), (27,13),
         (19,9), (13,20),
         (9,28), (9,4), (20,4),
         (28,7)
        ]
    E = [Edge(D[x],D[y]) for x,y in e]
    return (V,E)

@pytest.fixture
def sample_G07():
    v = range(1,24)
    V = [Vertex(x) for x in map(str,v)]
    D = dict(zip(v,V))
    e = [(1,13), (1,21), (1,4), (1,3), (2,3), (2,20),
         (3,4), (3,5), (3,23),
         (4,6), (5,7),
         (6,8), (6,16), (6,23), (7,9),
         (8,10), (8,11), (9,12),
         (10,13), (10,14), (10,15), (11,15), (11,16), (12,20),
         (13,17), (14,17), (14,18), (16,18), (16,19), (16,20),
         (18,21), (19,22),
         (21,23), (22,23)
        ]
    E = [Edge(D[x],D[y]) for x,y in e]
    return (V,E)

@pytest.fixture
def sample_G08():
    v = 'abc'
    V = [Vertex(x) for x in v]
    D = dict(zip(v,V))
    e = ['bc','ac']
    E = [Edge(D[xy[0]],D[xy[1]]) for xy in e]
    return (V,E)

