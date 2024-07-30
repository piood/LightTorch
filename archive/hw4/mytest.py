import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import torch

import needle as ndl
from needle import backend_ndarray as nd

np.random.seed(1)

_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def tt():
    print(device)
    
tt()