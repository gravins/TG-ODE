from xmlrpc.client import Boolean
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch.nn import Module, Linear, Identity
from torch_geometric.nn import TAGConv
from torch_geometric.data import Data
from typing import Optional


class TemporalGraphEuler(Module):
    '''
        h^l_v = h^(l-1)_v + e * activ_fun(-L X W)
    '''
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 K: int = 2,
                 normalization: bool = True, #Optional[str] = 'sym',
                 epsilon: float = 0.1,
                 activ_fun: Optional[str] = 'tanh',
                 use_previous_state: bool = False,
                 bias: bool = True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.K = K
        self.normalization = normalization
        self.activ_fun = getattr(torch, activ_fun) if activ_fun is not None else Identity()
        self.bias = bias
        self.input_time_dim = input_time_dim
        self.hidden_time_dim = hidden_time_dim
        self.use_previous_state = use_previous_state


        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim
    
        self.emb_t = None
        self.time_aggr = None
        if self.input_time_dim is not None:
            # Time encoder
            assert hidden_time_dim is not None, 'hidden_time_dim cannot be None when input_time_dim is not None'
            self.emb_t = Linear(self.input_time_dim, self.hidden_time_dim)
            
            assert time_aggr == 'concat' or time_aggr == 'add', f'time_aggr can be concat, add or None; not {time_aggr}'
            if time_aggr == 'concat':
                self.time_aggr = lambda x, y: torch.cat([x,y], dim=1)
                inp = inp + self.hidden_time_dim
            else:
                # Add by default
                assert inp == self.hidden_time_dim
                self.time_aggr = lambda x, y: x+y

        self.conv = TAGConv(in_channels = inp,
                            out_channels = inp,
                            K = self.K,
                            normalize = self.normalization,
                            bias = self.bias)

        if not readout: assert inp == self.output_dim, 'hidden_dim should be the same as output_dim when there is no readout'
        self.readout = Linear(inp, self.output_dim) if readout else None


    def forward(self, data: Data, prev_h: Optional[torch.Tensor]=None) -> torch.Tensor:
        x, edge_index, delta_t = data.x, data.edge_index, data.delta_t
        t_enc = data.t_enc if hasattr(data, 't_enc') else None 

        # Build (node, timestamp) encoding
        h = self.emb(x) if self.emb else x
        if self.emb_t:
            t_enc = self.emb_t(t_enc) 
            h = self.time_aggr(h, t_enc)

        if self.use_previous_state and prev_h is not None: 
            h = h + prev_h

        for _ in range(delta_t):
            conv = self.conv(h, edge_index)
            h = h + self.epsilon * self.activ_fun(conv)
            
        y = self.readout(h) if self.readout is not None else h
        return y, h
