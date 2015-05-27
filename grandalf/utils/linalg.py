from itertools import izip
from math import sqrt
from array import array as _array


def coerce_(types):
    if str in types: raise TypeError
    if complex in types: raise TypeError
    dtype = 'i'
    if long in types: dtype = 'l'
    if float in types: dtype = 'd'
    return dtype

constants = (int,long,float)

# minimalistic numpy.array replacement class used as fallback
# when numpy is not found in geometry module
class array(object):

    def __init__(self,data,dtype=None,copy=True):
        self.dim = len(data)
        if self.dim>0 and dtype is None:
            types = set([type(x) for x in data])
            dtype = coerce_(types)
        self.dtype = dtype
        if copy is True:
            self.data = _array(dtype,data)
        else:
            raise NotImplementedError

    def coerce(self,dtype):
        self.data = _array(dtype,self.data)
        self.dtype = dtype

    def __len__(self): return self.dim

    def __str__(self): return str(self.data)

    def __add__(self,v):
        if isinstance(v,constants):
            v = array([v]*self.dim)
        assert v.dim==self.dim
        return array([x+y for (x,y) in izip(self.data,v.data)])

    def __sub__(self,v):
        if isinstance(v,constants):
            v = array([v]*self.dim)
        assert v.dim==self.dim
        return array([x-y for (x,y) in izip(self.data,v.data)])

    def __neg__(self):
        return array([-x for x in self.data],dtype=self.dtype)

    def __radd__(self,v):
        return self+v
    def __rsub__(self,v):
        return (-self)+v

    def dot(self,v):
        assert v.dim==self.dim
        return sum([x*y for (x,y) in izip(self.data,v.data)])

    def __rmul__(self,k):
        return array([k*x for x in self.data])

    def __mul__(self,v):
        if isinstance(v,constants):
            v = array([v]*self.dim)
        assert v.dim==self.dim
        return array([x*y for (x,y) in zip(self.data,v.data)])

    def __div__(self,v):
        if isinstance(v,constants):
            v = array([v]*self.dim)
        assert v.dim==self.dim
        return array([x/y for (x,y) in zip(self.data,v.data)])

    def norm(self):
        return sqrt(self.dot(self))

    def __iter__(self):
        for x in self.data: yield x

#------------------------------------------------------------------------------
# minimalistic numpy.matrix replacement class used as fallback
# when numpy is not found in geometry module
class matrix(object):

    def __init__(self,data,dtype=None,copy=True):
        # check input data types:
        types = set([type(v) for v in data])
        if len(types)>1: raise TypeError
        t = types.pop()
        # import data:
        if t in (int,long,float):
            self.data = [array(data,dtype,copy)]
        else:
            self.data = [array(v,dtype,copy) for v in izip(*data)]
        # define matrix sizes:
        self.n = len(self.data)
        sizes = set([len(v) for v in self.data])
        if len(sizes)>1: raise ValueError
        self.p = sizes.pop()
        if dtype is None:
            # coerce types of arrays of matrix:
            types = set([v.dtype for v in self.data])
            dtype = coerce_(types)
            for v in self.data:
                v.coerce(dtype)
        self.dtype = dtype

    def __len__(self): return self.n

    def __str__(self): return '\n'.join([str(v) for v in self.data])

    @property
    def shape(self): return (self.n,self.p)

    def lvecs(self): return self.data

    def cvecs(self): return [array(v,self.dtype) for v in izip(*self.data)]

    def transpose(self):
        return matrix(self.data,dtype=self.dtype)

    def __add__(self,m):
        if isinstance(m,constants):
            return matrix([u+m for u in self.data])
        else:
            assert self.shape == m.shape
            return matrix([u+v for (u,v) in izip(self.data,m.data)])

    def __sub__(self,m):
        if isinstance(m,constants):
            return matrix([u-m for u in self.data])
        else:
            assert self.shape == m.shape
            return matrix([u-v for (u,v) in izip(self.data,m.data)])

    def __neg__(self):
        return matrix([-x for x in self.data],dtype=self.dtype)

    def __radd__(self,v):
        return self+v
    def __rsub__(self,v):
        return (-self)+v

    def __rmul__(self,k):
        if not isinstance(k,constants): raise TypeError
        return matrix([k*v for v in self.cvecs()])

    def __mul__(self,X):
        if isinstance(X,constants): raise TypeError
        if isinstance(X,array):
            assert X.dim==self.p
            return array([v.dot(X) for v in self.data])
        if isinstance(X,matrix):
            assert X.n == self.p
            return matrix([self*v for v in X.cvecs()])
