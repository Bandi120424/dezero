import os
import weakref
import numpy as np
import dezero.functions as F
from dezero import cuda
from dezero.core import Parameter
from dezero.utils import pair


# =============================================================================
# Layer (base class)
# =============================================================================
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)