.. grandalf documentation master file

======================
Grandalf documentation
======================

Grandalf is a python package made for experimentations with graphs and drawing
algorithms. It is written in pure python, and implements two layouts: the Sugiyama
hierarchical layout and the force-driven or energy minimization approach.
While not as fast or featured as _graphviz_ or other libraries like _OGDF_ (C++),
_GDToolkit_ (C), _tulip_ (Java), it provides a way to *draw* and *navigate* graphs
no larger than thousands of nodes, while keeping the source code simple enough
to make it possible to easily tweak and hack any part of it for experimental purpose.
With a total of about 1500 lines of python, the code involved in
drawing the Sugiyama (dot) layout fits in less than 600 lines.
The energy minimization approach is comprised of only 250 lines!

Grandalf does only two not-so-simple things:

    - computing the nodes (x,y) coordinates (based on provided nodes dimensions, and a
      chosen layout)
    - routing the edges with lines or nurbs

It doesn't depend on any GTK/Qt/whatever graphics toolkit.
This means that it will help you find *where* to
draw things like nodes and edges, but it's up to you to actually draw things with
your favorite graphics toolkit.
Take a look at amoco_ (amoco/ui/graphics/) to see how Grandalf can be used to
render some graphs on a Qt5 canvas.


.. _amoco: https://github.com/bdcht/amoco

.. ----------------------------------------------------------------------------  
.. _user-docs:

.. toctree::
   :maxdepth: 1
   :caption: User Documentation

   installation
   quickstart
   examples
   advanced

.. ----------------------------------------------------------------------------  
.. _devel-docs:

.. toctree::
   :maxdepth: 1
   :caption: Application Programming Interface

   overview
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

