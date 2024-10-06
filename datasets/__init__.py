from ._base import get_dataset, register_dataset
from .bmnist import BinaryMNIST
from .text8 import Text8Dataset

try:
    from .promoter import TSSDatasetS
except ImportError:
    print('[WARNING]: dependencies for TSSDatasetS not installed, skipping import')
