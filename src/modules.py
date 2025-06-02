import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP Wrapper class from https://github.com/junhongmit/FraudGT
    """
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None, num_layers=2, final_act = False, **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        layers.append(GeneralMultiLayer(num_layers - 1, dim_in, dim_inner, dim_inner, final_act=final_act))
        layers.append(Linear(dim_inner, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch

class GeneralMultiLayer(nn.Module):
    """
    General wrapper for stack of layers from https://github.com/junhongmit/FraudGT
    """
    def __init__(self, num_layers, dim_in, dim_out, dim_inner=None, final_act=True, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch

class GeneralLayer(nn.Module):
    """
    General wrapper for layers from https://github.com/junhongmit/FraudGT
    """
    def __init__(self, dim_in, dim_out, has_act=True, has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = CustomLinear(dim_in, dim_out, bias=True, **kwargs)
        layer_wrapper = []
        gnn_dropout = 0.2 # TODO: COULD CHANGE?

        if gnn_dropout > 0:
            layer_wrapper.append(nn.Dropout(p=gnn_dropout))
        if has_act:
            layer_wrapper.append(nn.ReLU())
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class CustomLinear(nn.Module):
    """
    A custom implementation of nn.Linear from https://github.com/junhongmit/FraudGT
    """
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(CustomLinear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch