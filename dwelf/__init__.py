import os

name = "dwelf"
__version__ = "1.0a1"
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = '{}/data/supermongo.mplstyle'.format(PACKAGEDIR)

try:
    __DWELF_SETUP__
except NameError:
    __DWELF_SETUP__ = False

if not __DWELF_SETUP__:
    from .model import *
