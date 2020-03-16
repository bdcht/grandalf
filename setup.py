from setuptools import setup

long_descr = '''
Grandalf is a python package made for experimentations with graphs drawing
algorithms. It is written in pure python, and currently implements two layouts:
the Sugiyama hierarchical layout and the force-driven or energy minimization approach.
While not as fast or featured as *graphviz* or other libraries like *OGDF* (C++),
it provides a way to walk and draw graphs
no larger than thousands of nodes, while keeping the source code simple enough
to tweak and hack any part of it for experimental purpose.
With a total of about 1500 lines of python, the code involved in
drawing the Sugiyama (dot) layout fits in less than 600 lines.
The energy minimization approach is comprised of only 250 lines!

Grandalf does only 2 not-so-simple things:

- computing the nodes (x,y) coordinates
  (based on provided nodes dimensions, and a chosen layout)
- routing the edges with lines or nurbs

It doesn't depend on any GTK/Qt/whatever graphics toolkit.
This means that it will help you find *where* to
draw things like nodes and edges, but it's up to you to actually draw things with
your favorite toolkit.
'''

setup(
    name='grandalf',
    version='0.7',

    description='Graph and drawing algorithms framework',
    long_description=long_descr,

    # The project's main homepage.
    url='https://github.com/bdcht/grandalf',

    # Author details
    author='Axel Tillequin',
    author_email='bdcht3@gmail.com',

    # Choose your license
    license='GPLv2 | EPLv1',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='graph drawings graphviz networkx',


    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['grandalf','grandalf/utils'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],

    install_requires=['pyparsing'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
      'full': ['numpy','ply'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
    },
)
