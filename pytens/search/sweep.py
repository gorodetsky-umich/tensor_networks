"""Locally sweep and refine structures"""

from abc import abstractmethod
import copy
import logging
import math
import random
from typing import Sequence

import networkx as nx

from pytens.algs import TreeNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.hierarchical.top_down.top_down import TopDownSearch
import pytens.search.hierarchical.top_down.white_box as wb
from pytens.search.utils import SearchResult, index_partition
from pytens.types import Index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



