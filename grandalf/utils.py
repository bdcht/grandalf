#!/usr/bin/env python
#
# This code is part of Grandalf
# Copyright (C) 2008 Axel Tillequin (bdcht3@gmail.com) and others
# published under GPLv2 license or EPLv1 license
# Contributor(s): Axel Tillequin, Fabio Zadrozny

from  numpy import array
from  math  import atan2,cos,sin,sqrt
import math, numpy

try:
    import ply.lex as lex
    import ply.yacc as yacc
    _has_ply = True
except ImportError:
    _has_ply = False

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
                print('warning: obj was already added in poset at index %d' \
                      %s.index(obj))
        self.s = s

    def __repr__(self):
        return 'poset(%r)' % (self.o,)

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
#  a random normalized n-dimension vector orthogonal to (1,1,1,...,1).
def  rand_ortho1(n):
    from random import SystemRandom
    r = SystemRandom()
    pos = [r.random() for x in xrange(n)]
    s = sum(pos)

    # Note: returning only positive values (if we had only negative values
    # an exception such as:
    # inf X:\grandalf\grandalf\layouts.py:784: RuntimeWarning: divide by zero encountered in double_scalars
    #   sfactor = 1.0/max(y.max(),x.max())
    # Could be found (randomly happening in test-layouts:test_splines).
    v = abs(array(pos,dtype=float)-(s/len(pos)))
    norm = sqrt(sum(v*v))
    return v/norm


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
    raise ValueError('no intersection found (point inside ?!). view: %s topt: %s' % (view, topt))


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


def median_wh(views):
    mw = [v.w for v in views]
    mh = [v.h for v in views]
    mw.sort()
    mh.sort()
    return (mw[len(mw)/2],mh[len(mh)/2])

#------------------------------------------------------------------------------
#  setcurve returns the spline curve that path through the list of points P.
#  The spline curve is a list of cubic bezier curves (nurbs) that have
#  matching tangents at their extreme points.
#  The method considered here is taken from "The NURBS book" (Les A. Piegl,
#  Wayne Tiller, Springer, 1997) and implements a local interpolation rather
#  than a global interpolation.
def setcurve(e,pts,tgs=None):
    P = map(array,pts)
    n = len(P)
    # tangent estimation
    if tgs:
      assert len(tgs)==n
      T = map(array,tgs)
      Q = [ P[k+1]-P[k] for k in range(0,n-1)]
    else:
      Q,T = tangents(P,n)
    splines=[]
    for k in xrange(n-1):
        t = T[k]+T[k+1]
        a = 16. - (t.dot(t))
        b = 12.*(Q[k].dot(t))
        c = -36. * Q[k].dot(Q[k])
        D = (b*b) - 4.*a*c
        assert D>=0
        sd = sqrt(D)
        s1,s2 = (-b-sd)/(2.*a),(-b+sd)/(2.*a)
        s = s2
        if s1>=0: s=s1
        C0 = tuple(P[k])
        C1 = tuple(P[k] + (s/3.)*T[k])
        C2 = tuple(P[k+1] -(s/3.)*T[k+1])
        C3 = tuple(P[k+1])
        splines.append([C0,C1,C2,C3])
    return splines

def tangents(P,n):
    assert n>=2
    Q = []
    T = []
    for k in xrange(0,n-1):
        q = P[k+1]-P[k]
        t = q/sqrt(q.dot(q))
        Q.append(q)
        T.append(t)
    T.append(t)
    return (Q,T)

#------------------------------------------------------------------------------
def setroundcorner(e,pts):
    P = map(array,pts)
    n = len(P)
    Q,T = tangents(P,n)
    c0 = P[0]
    t0 = T[0]
    k0 = 0
    splines = []
    k  = 1
    while k<n:
        z = abs(t0[0]*T[k][1]-(t0[1]*T[k][0]))
        if z<1.e-6:
            k+=1
            continue
        if (k-1)>k0: splines.append([c0,P[k-1]])
        if (k+1)<n:
            splines.extend(setcurve(e,[P[k-1],P[k+1]],tgs=[T[k-1],T[k+1]]))
        else:
            splines.extend(setcurve(e,[P[k-1],P[k]],tgs=[T[k-1],T[k]]))
            break
        if (k+2)<n:
            c0 = P[k+1]
            t0 = T[k+1]
            k0 = k+1
            k+=2
        else:
            break
    return splines or [[P[0],P[-1]]]

#------------------------------------------------------------------------------
# LALR(1) parser for Graphviz dot file format.
class Dot:

    _reserved = (
        'strict',
        'graph',
        'digraph',
        'subgraph',
        'node',
        'edge',
    )
    _tokens = (
        'regulars',
        'string',
        'html',
        'comment',
    )+_reserved

    _literals = [',',';','-','>','=',':','[',']','{','}']

    class Lexer(object):
        def __init__(self):
            self.whitespace = '\0\t\n\f\r '
            self.reserved = Dot._reserved
            self.tokens = Dot._tokens
            self.literals = Dot._literals
            self.t_ignore = self.whitespace

        def t_regulars(self,t):
            r'[-]?[\w.]+'
            v = t.value.lower()
            if v in self.reserved:
                t.type = v
                return t
            # check numeric string
            if v[0].isdigit() or v[0] in ['-','.']:
                try:
                    float(v)
                except ValueError:
                    print('invalid numeral token: %s'%v)
                    raise SyntaxError
            elif '.' in v: # forbidden in non-numeric
                raise SyntaxError
            return t

        def t_comment_online(self,t):
            r'(//(.*)\n)|\\\n'
            pass

        def t_comment_macro(self,t):
            r'(\#(.*)\n)'
            pass

        def t_comment_multline(self,t):
            r'(/\*)'
            start=t.lexer.lexpos
            t.lexer.lexpos = t.lexer.lexdata.index('*/',start)+2

        def t_string(self,t):
            r'"'
            start=t.lexer.lexpos-1
            i = t.lexer.lexdata.index('"',start+1)
            while t.lexer.lexdata[i-1] =='\\' :
                i = t.lexer.lexdata.index('"',i+1)
            t.value = t.lexer.lexdata[start:i+1]
            t.lexer.lexpos = i+1
            return t

        def t_html(self,t):
            r'<'
            start=t.lexer.lexpos-1
            level=1
            i=start+1
            while level>0:
                c = t.lexer.lexdata[i]
                if c=='<': level += 1
                if c=='>': level -= 1
                i += 1
            t.value = t.lexer.lexdata[start:i]
            t.lexer.lexpos = i
            return t

        def t_ANY_error(self,t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        def build(self,**kargs):
            if _has_ply:
                self._lexer = lex.lex(module=self, **kargs)

        def test(self,data):
            self._lexer.input(data)
            while 1:
                tok = self._lexer.token()
                if not tok:
                    break
                print(tok)

    # Classes for the AST returned by Parser:
    class graph(object):
        def __init__(self,name,data,strict=None,direct=None):
            self.name = name
            self.strict = strict
            self.direct = direct
            self.nodes = {}
            self.edges = []
            self.subgraphs = []
            self.attr = {}
            eattr = {}
            nattr = {}
            for x in data: # data is a statements (list of stmt)
                # x is a stmt, ie one of:
                # a graph object (subgraph)
                # a attr object (graph/node/edge attributes)
                # a dict object (ID=ID)
                # a node object
                # a list of edges
                if isinstance(x,Dot.graph):
                    self.subgraphs.append(x)
                elif isinstance(x,Dot.attr):
                    if x.type=='graph':
                        self.attr.update(x.D)
                    elif x.type=='node' :
                        nattr.update(x.D)
                    elif x.type=='edge' :
                        eattr.update(x.D)
                    else :
                        raise TypeError,'invalid attribute type'
                elif isinstance(x,dict):
                    self.attr.update(x)
                elif isinstance(x,Dot.node):
                    x.attr.update(nattr)
                    self.nodes[x.name] = x
                else:
                    for e in x:
                        e.attr.update(eattr)
                        self.edges.append(e)
                        for n in [e.n1,e.n2]:
                            if isinstance(n,Dot.graph): continue
                            if n.name not in self.nodes:
                                n.attr.update(nattr)
                                self.nodes[n.name] = n

        def __repr__(self):
            u = u'<%s instance at %x, name: %s, %d nodes>'%(
                   self.__class__,
                   id(self),
                   self.name,
                   len(self.nodes))
            return u.encode('utf-8')

    class attr(object):
        def __init__(self,type,D):
            self.type=type
            self.D = D

    class edge(object):
        def __init__(self,n1,n2):
            self.n1 = n1
            self.n2 = n2
            self.attr = {}

    class node(object):
        def __init__(self,name,port=None):
            self.name = name
            self.port = port
            self.attr = {}

    class Parser(object):
        def __init__(self):
            self.tokens = Dot._tokens

        def __makelist(self,p):
            N=len(p)
            if N>2:
                L = p[1]
                L.append(p[N-1])
            else:
                L = []
                if N>1:
                    L.append(p[N-1])
            p[0] = L

        def p_Data(self,p):
            '''Data : Data Graph
                    | Graph'''
            self.__makelist(p)

        def p_Graph_strict(self,p):
            '''Graph : strict graph name Block'''
            p[0] = Dot.graph(name=p[3],data=p[4],strict=1,direct=0)
            #print 'Dot.Parser: graph object %s created'%p[0].name

        def p_Graph_graph(self,p):
            '''Graph : graph name Block'''
            p[0] = Dot.graph(name=p[2],data=p[3],strict=0,direct=0)

        def p_Graph_strict_digraph(self,p):
            '''Graph : strict digraph name Block'''
            p[0] = Dot.graph(name=p[3],data=p[4],strict=1,direct=1)

        def p_Graph_digraph(self,p):
            '''Graph : digraph name Block'''
            p[0] = Dot.graph(name=p[2],data=p[3],strict=0,direct=1)

        def p_ID(self,p):
            '''ID : regulars
                  | string
                  | html '''
            p[0] = p[1]

        def p_name(self,p):
            '''name : ID
                    | '''
            if len(p)==1:
                p[0]=''
            else:
                p[0]=p[1]

        def p_Block(self,p):
            '''Block : '{' statements '}' '''
            p[0] = p[2]

        def p_statements(self,p):
            '''statements : statements stmt
                          | stmt
                          | '''
            self.__makelist(p)

        def p_stmt(self,p):
            '''stmt : stmt ';' '''
            p[0] = p[1]

        def p_comment(self,p):
            '''stmt : comment'''
            pass  # comment tokens are not outputed by lexer anyway

        def p_stmt_sub(self,p):
            '''stmt : sub'''
            p[0] = p[1]

        def p_subgraph(self,p):
            '''sub : subgraph name Block
                   | Block '''
            N = len(p)
            if N>2:
                ID = p[2]
            else:
                ID = ''
            p[0] = Dot.graph(name=ID,data=p[N-1],strict=0,direct=0)

        def p_stmt_assign(self,p):
            '''stmt : affect '''
            p[0] = p[1]

        def p_affect(self,p):
            '''affect : ID '=' ID '''
            p[0] = dict([(p[1],p[3])])

        def p_stmt_lists(self,p):
            '''stmt : graph attrs
                    | node  attrs
                    | edge  attrs '''
            p[0] = Dot.attr(p[1],p[2])

        def p_attrs(self,p):
            '''attrs : attrs attrl
                     | attrl '''
            if len(p)==3:
                p[1].update(p[2])
            p[0] = p[1]

        def p_attrl(self,p):
            '''attrl : '[' alist ']' '''
            L={}
            for a in p[2]:
                if isinstance(a,dict):
                    L.update(a)
                else:
                    L[a] = 'true'
            p[0] = L

        def p_alist_comma(self,p):
            '''alist : alist ',' alist '''
            p[1].extend(p[3])
            p[0] = p[1]

        def p_alist_affect(self,p):
            '''alist : alist affect
                     | alist ID
                     | affect
                     | ID
                     | '''
            self.__makelist(p)

        def p_stmt_E_attrs(self,p):
            '''stmt : E attrs '''
            for e in p[1]: e.attr = p[2]
            p[0] = p[1]

        def p_stmt_N_attrs(self,p):
            '''stmt : N attrs '''
            p[1].attr = p[2]
            p[0] = p[1]

        def p_stmt_EN(self,p):
            '''stmt : E
                    | N '''
            p[0] = p[1]

        def p_E(self,p):
            '''E : E   link
                 | elt link '''
            try:
                L = p[1]
                L.append(Dot.edge(L[-1].n2,p[2]))
            except:
                L = []
                L.append(Dot.edge(p[1],p[2]))
            p[0] = L

        def p_elt(self,p):
            '''elt : N
                   | sub '''
            p[0] = p[1]

        def p_link(self,p):
            '''link : '-' '>' elt
                    | '-' '-' elt '''
            p[0] = p[3]

        def p_N_port(self,p):
            '''N : ID port '''
            p[0] = Dot.node(p[1],port=p[2])

        def p_N(self,p):
            '''N : ID '''
            p[0] = Dot.node(p[1])

        def p_port(self,p):
            '''port : ':' ID '''
            p[0] = p[2]

        def p_port2(self,p):
            '''port : port port'''
            assert p[2] in ['n','ne',
                            'e','se',
                            's','sw',
                            'w','nw',
                            'c','_']
            p[0] = "%s:%s"%(p[1],p[2])

        def p_error(self,p):
            print('Syntax Error: %s' % (p,))
            self._parser.restart()

        def build(self,**kargs):
            opt=dict(debug=0,write_tables=0)
            opt.update(**kargs)
            if _has_ply:
                self._parser = yacc.yacc(module=self,**opt)

    def __init__(self,**kargs):
        self.lexer  = Dot.Lexer()
        self.parser = Dot.Parser()
        if not _has_ply:
            print('warning: Dot parser not supported (install python-ply)')

    def parse(self,data):
        try:
            self.parser._parser.restart()
        except AttributeError:
            self.lexer.build(reflags=lex.re.UNICODE)
            self.parser.build()
        except:
            print('unexpected error')
            return None
        try:
            s = data.decode('utf-8')
        except UnicodeDecodeError:
            s = data
        L=self.parser._parser.parse(s,
                                    lexer=self.lexer._lexer)
        return L

    def read(self,filename):
        f = file(filename,'rb') # As it'll try to decode later on with utf-8, read it binary at this point.
        return self.parse(f.read())


class Point(object):
    '''
    Helper class representing a point.
    '''

    def __init__(self, *pts):
        self.x, self.y = pts

    def __getitem__(self, i):
        if i == 0:
            return self.x

        if i == 1:
            return self.y

        raise AssertionError('For 2d point can only get 0 or 1 (trying to get: %s)' % (i,))

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.x
        yield self.y

    def distance(self, p2):
        x1, y1 = self
        x2, y2 = p2
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def __str__(self):
        return '(%s, %s)' % (self.x, self.y)

    def __repr__(self):
        return 'Point(%s, %s)' % (self.x, self.y)


def angle_between_360_degrees(angle):
    while angle >= 360.0:
        angle -= 360.0
    while angle < 0.0:
        angle += 360.0
    if angle == 360.0:
        angle = 0.0

    return angle


def angle_to_x_axis_in_degrees(pt0, pt1):
    base_p0 = (0.0, 0.0)
    base_p1 = (1.0, 0.0)

    line1 = (pt0[0] - pt1[0], pt0[1] - pt1[1])
    line2 = (base_p1[0] - base_p0[0], base_p1[1] - base_p0[1])

    x1 = float(line1[0])
    y1 = float(line1[1])
    x2 = float(line2[0])
    y2 = float(line2[1])
    divide_by = (abs(x1 * x2) + abs(y1 * y2))

    if divide_by == 0.0:
        if pt0[1] > pt1[1]:
            return 90.0
        else:
            return 270.0
    else:
        tg = ((x1 * y2) - (x2 * y1)) / divide_by
        tan_degrees = math.degrees(math.atan(tg))

        if pt0[0] > pt1[0]:
            tan_degrees = -tan_degrees
        else:
            tan_degrees += 180.0

        angle = tan_degrees
        angle = angle_between_360_degrees(angle)

        return angle

def new_point_at_distance(pt, distance, angle):
    angle = float(angle)
    x, y = pt[0], pt[1]
    x += float(distance) * numpy.cos(numpy.deg2rad(angle))
    y += float(distance) * numpy.sin(numpy.deg2rad(angle))
    return float(x), float(y)

