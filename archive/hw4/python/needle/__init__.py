from . import ops
from .ops import *
#from .autograd import Tensor, cpu, all_devices
from .autograd import Tensor
from . import init
from .init import ones, zeros, zeros_like, ones_like

from . import data
from . import nn
from . import optim
from .backend_selection import *
