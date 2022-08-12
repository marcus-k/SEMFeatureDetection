# -*- coding: utf-8 -*-
#
# Author: Marcus Kasdorf

from .finder import EllipseFinder

try:
    from .AAMED import AAMED
except ImportError as e:
    print("Error importing AAMED (does not exist or invalid format). Continuing without AAMED.")

from .basic import Basic
from .canny import Canny
from .EM import EM
