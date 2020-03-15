import pytest
from grandalf.utils.linalg import *

def test_linalg_001():
    assert coerce_(None) == ('i',int)
    assert coerce_([int,long]) == ('l',long)
    assert coerce_([int,float]) == ('d',float)
    assert coerce_([long,float]) == ('d',float)
    with pytest.raises(TypeError) as x:
        coerce_([complex])
        assert x.type==TypeError

def test_linalg_001():
    v1 = array([1,2,3])
    assert v1.dim==3
    assert v1.typecode=='i'
    assert v1.dtype==int
    assert len(v1)==3
    assert v1[1]==2
    assert (v1+v1)[1]==4
    assert (2*v1)[0]==2
    assert (v1*2)[2]==6
    assert sum(-v1+v1)==0
    assert v1.dot(v1) == 14
    assert (v1//v1).max()==1
    assert (v1//v1).min()==1
    assert v1.transpose().shape == (1,3)


def test_linalg_002():
    v1 = array([1,2.,3])
    assert v1.typecode=='d'
    assert type(v1[0])==float
    assert v1[0]==1.
    assert float(v1.transpose()*v1)==v1.dot(v1)
    m1 = matrix(v1)
    assert float(m1*v1) == v1.dot(v1)
    assert (2*m1)[0,1] == (m1+m1)[0,1]
    assert (m1*2).shape == (2*m1).shape

def test_linalg_003():
    m1 = matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    assert m1.shape == (3,4)
    assert len(m1)==3*4
    m2 = matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12]],transpose=True)
    assert m2.shape == (4,3)
    assert m1.transpose()[1,2] == m2[1,2]
    assert m1[1:,1:].shape == (2,3)
    v = m2[1:,0]
    assert isinstance(v,array)
    l2 = m2[1,1:]
    assert isinstance(l2,matrix)
    assert l2.n==1
    assert m2[0,0]==1
