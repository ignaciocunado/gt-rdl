from functools import partial

import torch
import torch.nn as nn
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_act


# Add Gaussian Error Linear Unit (GELU).
register_act('relu', nn.ReLU(inplace=cfg.mem.inplace))
register_act('gelu', nn.GELU())
register_act('elu', nn.ELU(inplace=cfg.mem.inplace))