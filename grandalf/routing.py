#!/usr/bin/env python
#
# This code is part of Grandalf
#  Copyright (C) 2011 Axel Tillequin (bdcht3@gmail.com)
# published under GPLv2 license

#  Edge routing algorithms.
#  These are mosty helpers for routing an edge 'e' through
#  points pts with various tweaks like moving the starting point
#  to the intersection with the bounding box and taking some constraints
#  into account, and/or moving the head also to its prefered position.
#  Of course, since gandalf only works with bounding boxes, the exact
#  shape of the nodes are not known and the edge drawing inside the bb
#  shall be performed by the drawing engine associated with 'views'.
#  (e.g. look at intersectC when the node shape is a circle)

from  utils import intersectR,getangle,setcurve,setroundcorner

#------------------------------------------------------------------------------
class  EdgeViewer(object):
    def setpath(self,pts):
        self._pts = pts

#------------------------------------------------------------------------------
#  basic edge routing with lines : nothing to do for routing
#  since the layout engine has already provided to list of points through which
#  the edge shall be drawn. We just compute the position where to adjust the
#  tail and head. 
def  route_with_lines(e,pts):
    assert hasattr(e,'view')
    tail_pos = intersectR(e.v[0].view,topt=pts[1])
    head_pos = intersectR(e.v[1].view,topt=pts[-2])
    pts[0]  = tail_pos
    pts[-1] = head_pos
    e.view.head_angle = getangle(pts[-2],pts[-1])

#------------------------------------------------------------------------------
#  enhanced edge routing where 'corners' of the above polyline route are
#  rounded with a bezier curve.
def route_with_splines(e,pts):
    route_with_lines(e,pts)
    splines = setroundcorner(e,pts)
    e.view.splines = splines

