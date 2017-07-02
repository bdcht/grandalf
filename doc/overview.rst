Overview
========

Grandalf is composed of 3 modules :mod:`graphs`, :mod:`layouts`,
:mod:`routing` and :ref:`utils` sub-package.
The `Vertex`, `Edge` and `Graph` classes implemented in modules
:mod:`graphs` are related to the mathematical representation
of a graph, including some important features like Dijkstra or
Tarjan algorithms.

The :mod:`layouts` module essentially implements :class:`SugiyamaLayout`
and :class:`DigcoLayout` which operate on a graph connex component.

The :mod:`routing` module deals with specialized edge-routing algorithms
and provides functions that can be used by layout classes during
the edges drawing phase.

modules
=======

.. automodule:: grandalf.graphs
    :members:

.. automodule:: grandalf.layouts
   :members:

.. automodule:: grandalf.routing
   :members:
