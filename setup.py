#!/usr/bin/env python

import sys

v = sys.version_info
if v.major!=2:
    print 'grandalf requires python 2.x'
    sys.exit(1)
if v.minor<7:
    print 'grandalf requires python>=2.7'
    sys.exit(1)

from distutils.core import setup

setup(
    name = 'Grandalf',
    version = '0.555',
    packages=['grandalf','grandalf/utils'],
    # Metadata
    author = 'Axel Tillequin',
    author_email = 'bdcht3@gmail.com',
    description = 'Dynamic 2D graph placement library',
    license = 'GPLv2 | EPLv1',
    keywords = 'graphs',
    url = 'https://github.com/bdcht/grandalf',
)
