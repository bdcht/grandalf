#!/usr/bin/env python
#
# This code is part of Grandalf
# Copyright (C) 2008 Axel Tillequin (bdcht3@gmail.com) 
# published under GPLv2 license

from  numpy import array,matrix,linalg
from  math  import atan2,cos,sin

#------------------------------------------------------------------------------
class  Poset(object):

    def __init__(self,L):
        self.o = []
        s = set()
        for obj in L:
            if not obj in s:
                self.o.append(obj)
                s.add(obj)
            else:
                print 'warning: obj was already added in poset at index %d' \
                      %s.index(obj)
        self.s = s

    def __repr__(self):
        return 'poset'+repr(self.o)
    def __str__(self):
        f='%%%dd'%len(str(len(self.o)))
        s=[]
        for i,x in enumerate(self.o):
            s.append(f%i+'.| %s'%repr(x))
        return '\n'.join(s)

    def add(self,obj):
        if obj not in self.s:
            self.o.append(obj)
            self.s.add(obj)

    def remove(self,obj):
        if obj in self.s:
            self.o.remove(obj)
            self.s.remove(obj)

    def index(self,obj):
        return self.o.index(obj)

    def __len__(self):
        return len(self.o)

    def __repr__(self):
        return repr(self.o)

    def __iter__(self):
        for obj in self.o:
            yield obj

    def __cmp__(self,other):
        return cmp(other.s,self.s)

    def __eq__(self,other):
        return other.s==self.s

    def __ne__(self,other):
        return other.s<>self.s

    def copy(self):
        return Poset(self.o)

    __copy__ = copy
    def deepcopy(self):
        from copy import deepcopy
        L = deepcopy(self.o)
        return Poset(L)

    def __or__(self,other):
        return self.union(other)

    def union(self,other):
        return Poset(self.o+other.o)

    def update(self,other):
        for obj in other:
            if obj not in self:
                self.add(obj)

    def __and__(self,other):
        return self.intersection(other)

    def intersection(self,other):
        return Poset(self.s.intersection(other.s))

    def __xor__(self,other):
        return self.symmetric_difference(other)

    def symmetric_difference(self,other):
        return Poset(self.s^other.s)

    def __sub__(self,other):
        return self.difference(other)

    def difference(self,other):
        return Poset(self.s-other.s)

    def __contains__(self,obj):
        return (obj in self.s)

    def issubset(self,other):
        return (self.s.issubset(other.s))

    def issuperset(self,other):
        return self.s.issuperset(other.s)

    __le__ = issubset
    __ge__ = issuperset

    def __lt__(self,other):
        return (self<=other and len(self)<>len(other))

    def __gt__(self,other):
        return (self>=other and len(self)<>len(other))


#  rand_ortho1 returns a numpy.array representing
#  a random normalized n-dimension vector orthogonal to 1.
def  rand_ortho1(n):
    from random import SystemRandom
    r = SystemRandom()
    pos = [r.random() for x in xrange(n)]
    s = sum(pos)
    return array(pos,dtype=float)-(s/len(pos))
  

#------------------------------------------------------------------------------
#TODO:  this was imported here from masr, but since we have
#  here access to numpy.array, we could use it for vectors operations.
def  intersect2lines((x1,y1),(x2,y2),(x3,y3),(x4,y4)):
    b = (x2-x1,y2-y1)
    d = (x4-x3,y4-y3)
    det = b[0]*d[1] - b[1]*d[0]
    if det==0: return None
    c = (x3-x1,y3-y1)
    t = float(c[0]*b[1] - c[1]*b[0])/(det*1.)
    if (t<0. or t>1.): return None
    t = float(c[0]*d[1] - c[1]*d[0])/(det*1.)
    if (t<0. or t>1.): return None
    x = x1 + t*b[0]
    y = y1 + t*b[1]
    return (x,y)


#------------------------------------------------------------------------------
#  intersectR returns the intersection point between the Rectangle
#  (w,h) that characterize the view object and the line that goes
#  from the views' object center to the 'topt' point.
def  intersectR(view,topt):
    # we compute intersection in local views' coord:
    # center of view is obviously :
    x1,y1 = 0,0
    # endpoint in view's coord:
    x2,y2 = topt[0]-view.xy[0],topt[1]-view.xy[1]
    # bounding box:
    bbx2 = view.w/2
    bbx1 = -bbx2
    bby2 = view.h/2
    bby1 = -bby2
    # all 4 segments of the bb:
    S = [((x1,y1),(x2,y2),(bbx1,bby1),(bbx2,bby1)),
              ((x1,y1),(x2,y2),(bbx2,bby1),(bbx2,bby2)),
              ((x1,y1),(x2,y2),(bbx1,bby2),(bbx2,bby2)),
              ((x1,y1),(x2,y2),(bbx1,bby2),(bbx1,bby1))]
    # check intersection with each seg:
    for segs in S:
        xy = intersect2lines(*segs)
        if xy!=None:
            x,y = xy
            # return global coord:
            x += view.xy[0]
            y += view.xy[1]
            return (x,y)
    # there can't be no intersection unless the endpoint was
    # inside the bb !
    raise ValueError,'no intersection found (point inside ?!)'


#------------------------------------------------------------------------------
def  getangle(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    theta = atan2(y2-y1,x2-x1)
    return theta


#------------------------------------------------------------------------------
#  intersectC returns the intersection point between the Circle
#  of radius r and centered on views' position with the line 
#  to the 'topt' point.
def  intersectC(view, r, topt):
    theta = getangle(view.xy,topt)
    x = int(cos(theta)*r)
    y = int(sin(theta)*r)
    return (x,y)


#------------------------------------------------------------------------------
#  setcontrols find the bezier control points associated to the rhs vector:
def  setcontrols(rhs):
    b=2.0
    n=len(rhs)
    x=range(n)
    t=range(n)
    x[0]=rhs[0]/b
    for i in range(1,n-1):
        t[i]=1./b
        b = 4.0-t[i]
        x[i] = (rhs[i]-x[i-1])/b
    t[n-1]=1./b
    b = 3.5-t[n-1]
    x[n-1]=(rhs[n-1]-x[n-2])/b
    for i in range(1,n):
        x[n-i-1] -= t[n-i]*x[n-i]
    return x


#------------------------------------------------------------------------------
#  setcurve returns the spline curve that path through the list of points P.
#  The spline curve is a list of cubic bezier curves (nurbs) that have
#  matching tangents at their extreme points.
def  setcurve(P):
    n = len(P)-1
    assert n>0
    if n==1:
        P0=P[0]
        P1=P[1]
        C1=((2*P0[0]+P1[0])/3.,(2*P0[1]+P1[1])/3.)
        C2=(2*C1[0]-P0[0],2*C1[1]-P0[1])
        return [[P0,C1,C2,P1]]
    else:
        rhs = range(n)
        rhs[0] = P[0][0]+2.*P[1][0]
        for i in range(1,n-1):
            rhs[i]=4*P[i][0]+2*P[i+1][0]
        rhs[n-1] = (8*P[n-1][0]+P[n][0])/2.
        x = setcontrols(rhs)
        rhs[0] = P[0][1]+2.*P[1][1]
        for i in range(1,n-1):
            rhs[i]=4*P[i][1]+2*P[i+1][1]
        rhs[n-1] = (8*P[n-1][1]+P[n][1])/2.
        y = setcontrols(rhs)
        C1 = zip(x,y)
        C2 = []
        for i in range(0,n-1):
            C2.append((2*P[i+1][0]-x[i+1],2*P[i+1][1]-y[i+1]))
        C2.append(((P[n][0]+x[n-1])/2.,(P[n][1]+y[n-1])/2.))
        splines=[]
        for i in range(n):
            splines.append([P[i],C1[i],C2[i],P[i+1]])
        return splines
