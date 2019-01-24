import os

name = "dwelf"
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = '{}/data/supermongo.mplstyle'.format(PACKAGEDIR)

from .model import *
