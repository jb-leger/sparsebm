__title__ = "sparsebm"
__author__ = "Gabriel Frisch"
__licence__ = "MIT"

version_info = (1, 3)
__version__ = ".".join(map(str, version_info))

from .lbm import LBM
from .sbm import SBM
from .utils import CARI
from .model_selection import ModelSelection
from .graph_generator import generate_LBM_dataset, generate_SBM_dataset

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("sparsebm.log")
ch = logging.StreamHandler()
fh.setLevel(logging.DEBUG)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
formatter_file = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
fh.setFormatter(formatter_file)
logger.addHandler(ch)
logger.addHandler(fh)
