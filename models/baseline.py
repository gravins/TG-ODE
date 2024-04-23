import torch
from torch.nn import Module, Linear, Identity
from torch_geometric.data import Data
from typing import Optional

import torch
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, GConvGRU, GConvLSTM, A3TGCN
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import GCNConv


class SpatioTemporalModel(Module):
    def __init__(self,                  
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 activ_fun: Optional[str] = 'tanh',
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 iterate: bool = False
                 ):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activ_fun = getattr(torch, activ_fun) if activ_fun is not None else Identity()
        self.input_time_dim = input_time_dim
        self.hidden_time_dim = hidden_time_dim
        self.time_aggr = time_aggr
        self.readout = readout
        self.iterate = iterate

        self.inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            self.inp = self.hidden_dim

        self.emb_t = None
        self.time_aggr = None
        if self.input_time_dim is not None:
            # Time encoder
            assert hidden_time_dim is not None, 'hidden_time_dim cannot be None when input_time_dim is not None'
            self.emb_t = Linear(self.input_time_dim, self.hidden_time_dim)
            
            assert time_aggr == 'concat' or time_aggr == 'add', f'time_aggr can be concat, add or None; not {time_aggr}'
            if time_aggr == 'concat':
                self.time_aggr = lambda x, y: torch.cat([x,y], dim=1)
                self.inp = self.inp + self.hidden_time_dim
            else:
                # Add by default
                assert self.inp == self.hidden_time_dim
                self.time_aggr = lambda x, y: x+y

        self.conv = None

        if not readout: assert self.inp == self.output_dim, 'hidden_dim should be the same as output_dim when there is no readout'
        self.readout = Linear(self.inp, self.output_dim) if readout else None

    def forward(self, data: Data, prev_h: Optional[torch.Tensor]=None) -> torch.Tensor:
        assert self.conv, 'The temporal graph encoder is not initialized'

        x, edge_index, delta_t = data.x, data.edge_index, data.delta_t
        t_enc = data.t_enc if hasattr(data, 't_enc') else None 

        # Build (node, timestamp) encoding
        h = self.emb(x) if self.emb else x
        if self.emb_t:
            t_enc = self.emb_t(t_enc) 
            h = self.time_aggr(h, t_enc)

        if not self.iterate:
            delta_t = 1
        
        for _ in range(delta_t):  
            h_state = self.conv(h, edge_index, H=prev_h)
            h = self.activ_fun(h_state)
            prev_h = h_state

        y = self.readout(h) if self.readout is not None else h
        return y, h_state


class DCRNNModel(SpatioTemporalModel):
    def __init__(self,                  
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 activ_fun: Optional[str] = 'tanh',
                 K: int = 2,
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 iterate: bool = False
                 ):

        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, 
                         activ_fun=activ_fun, input_time_dim=input_time_dim, iterate = iterate,
                         hidden_time_dim=hidden_time_dim, time_aggr=time_aggr,  readout=readout)

        self.K = K
        self.conv = DCRNN(in_channels = self.inp,
                           out_channels = self.inp,
                           K = self.K)


class TGCNModel(SpatioTemporalModel):
    def __init__(self,                  
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 activ_fun: Optional[str] = 'tanh',
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 iterate: bool = False
                 ):

        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, 
                         activ_fun=activ_fun, input_time_dim=input_time_dim, iterate = iterate, 
                         hidden_time_dim=hidden_time_dim, time_aggr=time_aggr,  readout=readout)

        self.conv = TGCN(in_channels = self.inp,
                          out_channels = self.inp)


class GCRN_LSTM_Model(SpatioTemporalModel):
    def __init__(self,                  
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 activ_fun: Optional[str] = 'tanh',
                 K: int = 2,
                 normalization: Optional[str] = None,
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 iterate: bool = False
                 ):

        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, 
                         activ_fun=activ_fun, input_time_dim=input_time_dim, iterate = iterate,
                         hidden_time_dim=hidden_time_dim, time_aggr=time_aggr,  readout=readout)
        
        self.K = K
        self.normalization = normalization
        
        self.conv = GConvLSTM(in_channels = self.inp,
                               out_channels = self.inp,
                               K = self.K,
                               normalization = self.normalization)

    def forward(self, data: Data, prev_h: Optional[torch.Tensor]=None) -> torch.Tensor:
        assert self.conv, 'The temporal graph encoder is not initialized'

        x, edge_index, delta_t = data.x, data.edge_index, data.delta_t
        t_enc = data.t_enc if hasattr(data, 't_enc') else None 

        # Build (node, timestamp) encoding
        h = self.emb(x) if self.emb else x
        if self.emb_t:
            t_enc = self.emb_t(t_enc) 
            h = self.time_aggr(h, t_enc)

        if prev_h is None:
            prev_H, prev_C = None, None
        else:
            prev_H, prev_C = prev_h
        
        _, edge_weight = get_laplacian(edge_index, normalization=self.normalization)
        
        if not self.iterate:
            delta_t = 1

        for _ in range(delta_t):
            H, C = self.conv(h, edge_index, H=prev_H, C=prev_C, lambda_max=edge_weight.max())
            h = self.activ_fun(H)
            prev_H = H
            prev_C = C

        y = self.readout(h) if self.readout is not None else h
        return y, torch.stack((H, C))
        

class GCRN_GRU_Model(SpatioTemporalModel):
    def __init__(self,                  
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 activ_fun: Optional[str] = 'tanh',
                 K: int = 2,
                 normalization: Optional[str] = None,
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 iterate: bool = False
                 ):

        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, 
                         activ_fun=activ_fun, input_time_dim=input_time_dim, iterate = iterate, 
                         hidden_time_dim=hidden_time_dim, time_aggr=time_aggr,  readout=readout)
        
        self.K = K
        self.normalization = normalization
        
        self.conv = GConvGRU(in_channels = self.inp,
                              out_channels = self.inp,
                              K = self.K,
                              normalization = self.normalization)

    def forward(self, data: Data, prev_h: Optional[torch.Tensor]=None) -> torch.Tensor:
        assert self.conv, 'The temporal graph encoder is not initialized'

        x, edge_index, delta_t = data.x, data.edge_index, data.delta_t
        t_enc = data.t_enc if hasattr(data, 't_enc') else None 

        # Build (node, timestamp) encoding
        h = self.emb(x) if self.emb else x
        if self.emb_t:
            t_enc = self.emb_t(t_enc) 
            h = self.time_aggr(h, t_enc)

        _, edge_weight = get_laplacian(edge_index, normalization=self.normalization)

        if not self.iterate:
            delta_t = 1

        for _ in range(delta_t):
            h_state = self.conv(h, edge_index, H=prev_h, lambda_max=edge_weight.max())
            h = self.activ_fun(h_state)
            prev_h = h_state

        y = self.readout(h) if self.readout is not None else h
        return y, h_state
    

class A3TGCNModel(SpatioTemporalModel):
    def __init__(self,                  
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 activ_fun: Optional[str] = 'tanh',
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 iterate: bool = False
                 ):

        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, 
                         activ_fun=activ_fun, input_time_dim=input_time_dim, iterate = iterate,
                         hidden_time_dim=hidden_time_dim, time_aggr=time_aggr,  readout=readout)

        self.conv = A3TGCN(in_channels = self.inp,
                            out_channels = self.inp,
                            periods = 1)

    def forward(self, data: Data, prev_h: Optional[torch.Tensor]=None) -> torch.Tensor:
        assert self.conv, 'The temporal graph encoder is not initialized'

        x, edge_index, delta_t = data.x, data.edge_index, data.delta_t
        t_enc = data.t_enc if hasattr(data, 't_enc') else None 

        # Build (node, timestamp) encoding
        h = self.emb(x) if self.emb else x
        if self.emb_t:
            t_enc = self.emb_t(t_enc) 
            h = self.time_aggr(h, t_enc)

        if not self.iterate:
            delta_t = 1

        for _ in range(delta_t):
            h_state = self.conv(h.view(h.shape[0], h.shape[1], 1), edge_index, H=prev_h) # A3TGCN input must have size [num_nodes, num_features, num_periods]
            h = self.activ_fun(h_state)
            prev_h = h_state
            
        y = self.readout(h) if self.readout is not None else h
        return y, h_state


from torchdyn.core import NeuralODE as tdNeuralODE
class NODE_odefunc(Module):
    def __init__(self, input_size, act) -> None:
        super().__init__()
        self.conv = torch.nn.Linear(input_size, input_size)
        self.act = act

    def forward(self, t, x):  # the t param is needed by the ODE solver.
        return self.act(self.conv(x))

class NeuralODE(SpatioTemporalModel):
    def __init__(self,                 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 epsilon: float = 0.1,
                 activ_fun: Optional[str] = 'tanh',
                 use_previous_state: bool = False,
                 torchdyn_method: bool = False) -> None:
        
        super().__init__(input_dim=input_dim, output_dim=output_dim, 
                         hidden_dim=hidden_dim, input_time_dim=input_time_dim, 
                         hidden_time_dim=hidden_time_dim, time_aggr=time_aggr, 
                         readout=readout, activ_fun=activ_fun)

        self.epsilon = epsilon
        self.use_previous_state = use_previous_state
        self.torchdyn_method = torchdyn_method
        
        if self.torchdyn_method:
            self.func = NODE_odefunc(self.inp, self.activ_fun)
            self.conv = tdNeuralODE(self.func, sensitivity='adjoint', 
                                    solver='euler', solver_adjoint='euler', 
                                    return_t_eval=False)
        else:
            self.conv = torch.nn.Linear(self.inp, self.inp)

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

        if self.torchdyn_method:
            # Employ torchdyn method
            t_span = [0.]
            for _ in range(delta_t):
                t_span.append(t_span[-1] + self.epsilon)
            self.t_span = torch.tensor(t_span) # the evaluation timesteps

            h = self.conv(h, t_span=self.t_span)
            h = h[-1] # conv returns node states at each evaluation step
        else:
            for _ in range(delta_t):
                h = h + self.epsilon * self.activ_fun(self.conv(h))
            
        y = self.readout(h) if self.readout is not None else h
        return y, h
    

class NDCN_odefunc(Module):
    def __init__(self, input_size, act) -> None:
        super().__init__()
        self.conv = GCNConv(
            in_channels = input_size, 
            out_channels = input_size
        )
        self.act = act
        self.edge_index = None

    def forward(self, t, x):  # the t param is needed by the ODE solver.
        return self.act(self.conv(x, self.edge_index))

class NDCN(SpatioTemporalModel):
    def __init__(self,                 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 input_time_dim: Optional[int] = None, # The dimension of time feature vector
                 hidden_time_dim: Optional[int] = None, # The dimension of time feature vector
                 time_aggr: Optional[str] = None, # How to aggregate time and hidden state 
                 readout: bool = True,
                 epsilon: float = 0.1,
                 activ_fun: Optional[str] = 'tanh',
                 cached: bool = False,
                 torchdyn_method: bool = False) -> None:
        
        super().__init__(input_dim=input_dim, output_dim=output_dim, 
                         hidden_dim=hidden_dim, input_time_dim=input_time_dim, 
                         hidden_time_dim=hidden_time_dim, time_aggr=time_aggr, 
                         readout=readout, activ_fun=activ_fun)
        
        self.epsilon = epsilon
        self.torchdyn_method = torchdyn_method
        self.cached = cached

        if self.torchdyn_method:
            self.func = NDCN_odefunc(self.inp, self.activ_fun)
            self.conv = tdNeuralODE(self.func, sensitivity='adjoint', 
                                    solver='euler', solver_adjoint='euler', 
                                    return_t_eval=False)
        else:
            self.conv = GCNConv(self.inp, self.inp, cached=self.cached)

    def forward(self, data: Data, prev_h: Optional[torch.Tensor]=None) -> torch.Tensor:

        x, edge_index, delta_t = data.x, data.edge_index, data.delta_t
        t_enc = data.t_enc if hasattr(data, 't_enc') else None 

        # Build (node, timestamp) encoding
        if prev_h is not None:
            h = prev_h
        else:
            # x is used only on the very first step
            h = self.emb(x) if self.emb else x 
            if self.emb_t:
                t_enc = self.emb_t(t_enc) 
                h = self.time_aggr(h, t_enc)

        if self.torchdyn_method:
            # Employ torchdyn method
            t_span = [0.]
            for _ in range(delta_t):
                t_span.append(t_span[-1] + self.epsilon)
            self.t_span = torch.tensor(t_span) # the evaluation timesteps

            if (not self.cached) or self.func.edge_index is None:
                self.func.edge_index = edge_index

            h = self.conv(h, t_span=self.t_span)
            h = h[-1] # conv returns node states at each evaluation step
        else:
            for _ in range(delta_t):
                h = h + self.epsilon * self.activ_fun(self.conv(h, edge_index))
            
        y = self.readout(h) if self.readout is not None else h
        return y, h