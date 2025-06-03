from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class OptionSide(Enum):
    CALL = 'CALL'
    PUT  = 'PUT'

class Option(ABC):
    """
    Abstract base class for financial options.
    """