from models import *
from numgraph import simple_grid_coo
from datasets import diffusion_functions, traffic_ablation_names


def get_conf(input_dim, output_dim, time_dim):
    for lr in [1e-2, 1e-3, 1e-4]:
        for wd in [1e-2, 1e-3]:
            for epsilon in [0.001, 0.01, 0.1, 0.5, 1.]:
                for hidden in [64, 32] if time_dim is not None else [None, 8]:
                    for activ_fun in ['tanh', 'relu', None]:
                        for time_aggr in ['concat', 'add'] if time_dim is not None else [None]:
                            if time_dim is not None:
                                t_hiddens = [time_dim // 2] if time_aggr == 'concat' else [hidden]
                            else:
                                t_hiddens = [None]
                            for t_hidden in t_hiddens:
                                for norm in [True, False]:
                                    for K in [1,2] if time_dim is not None else [5]:
                                        for use_previous_state in [True, False]:
                                            readout = hidden != None
                                            yield {
                                                'model': {
                                                    'input_dim': input_dim,
                                                    'output_dim': output_dim,
                                                    'hidden_dim': hidden,
                                                    'input_time_dim': time_dim,
                                                    'hidden_time_dim': t_hidden,
                                                    'time_aggr': time_aggr,
                                                    'readout': readout,
                                                    'K': K,
                                                    'normalization': norm,
                                                    'epsilon': epsilon,
                                                    'activ_fun': activ_fun,
                                                    'use_previous_state': use_previous_state,
                                                    'bias': True
                                                },
                                                'optim': {
                                                    'lr': lr,
                                                    'weight_decay': wd
                                                }
                                            }

def get_conf_ablation(input_dim, output_dim, time_dim):
    yield {
        'model': {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': 8,
            'input_time_dim': time_dim,
            'hidden_time_dim': None,
            'time_aggr': None,
            'readout': True,
            'K': 5,
            'normalization': False,
            'epsilon': 0.5,
            'activ_fun': 'tanh',
            'use_previous_state': False,
            'bias': True
        },
        'optim': {
            'lr': 0.0001,
            'weight_decay': 0.001
        }
    }


def get_conf_baseline(model_name, input_dim, output_dim, time_dim):
    for lr in [1e-2, 1e-3, 1e-4]:
        for wd in [1e-2, 1e-3]:
            for hidden in [64, 32] if time_dim is not None else [None, 8]:
                for activ_fun in ['tanh', 'relu', None]:
                    for time_aggr in ['concat', 'add'] if time_dim is not None else [None]:
                        if time_dim is not None:
                            t_hiddens = [time_dim // 2] if time_aggr == 'concat' else [hidden]
                        else:
                            t_hiddens = [None]
                        for t_hidden in t_hiddens:
                            for iterate in [True, False]:
                                readout = hidden != None
                                conf =  {
                                    'model': {
                                        'input_dim': input_dim,
                                        'output_dim': output_dim,
                                        'hidden_dim': hidden,
                                        'activ_fun': activ_fun,
                                        'input_time_dim': time_dim,
                                        'hidden_time_dim': t_hidden,
                                        'time_aggr': time_aggr,
                                        'readout': readout,
                                        'iterate': iterate,
                                    },
                                    'optim': {
                                        'lr': lr,
                                        'weight_decay': wd
                                    }
                                }
                                if model_name in ['DCRNN', 'GCRN_LSTM', 'GCRN_GRU']:
                                    for k in [1,2] if time_dim is not None else [2,5]:
                                        conf['model']['K'] = k
                                        if model_name in ['GCRN_LSTM', 'GCRN_GRU']:
                                            for norm in ['sym']: #, None, 'rw']:
                                                conf['model']['normalization'] = norm
                                                yield conf
                                        else:
                                            yield conf
                                else:
                                    assert model_name in ['TGCN', 'A3TGCN']
                                    yield conf


def get_conf_node(model_name, input_dim, output_dim, time_dim):
    for lr in [1e-2, 1e-3, 1e-4]:
        for wd in [1e-2, 1e-3]:
            for epsilon in [0.001, 0.01, 0.1, 0.5, 1.]:
                for hidden in [64, 32] if time_dim is not None else [None, 8]:
                    for activ_fun in ['tanh', 'relu', None]:
                        for time_aggr in ['concat', 'add'] if time_dim is not None else [None]:
                            if time_dim is not None:
                                t_hiddens = [time_dim // 2] if time_aggr == 'concat' else [hidden]
                            else:
                                t_hiddens = [None]
                            for t_hidden in t_hiddens:
                                readout = hidden != None
                                for torchdyn_method in [True, False]:
                                    conf =  {
                                        'model': {
                                            'input_dim': input_dim,
                                            'output_dim': output_dim,
                                            'hidden_dim': hidden,
                                            'activ_fun': activ_fun,
                                            'input_time_dim': time_dim,
                                            'hidden_time_dim': t_hidden,
                                            'time_aggr': time_aggr,
                                            'readout': readout,
                                            'epsilon': epsilon,
                                            'torchdyn_method': torchdyn_method
                                        },
                                        'optim': {
                                            'lr': lr,
                                            'weight_decay': wd
                                        }
                                    }
                                    if model_name == 'NDCN':
                                        conf['model']['cached'] = True
                                        yield conf
                                    elif model_name == 'NODE':
                                        for use_previous_state in [True, False]:
                                            conf['model']['use_previous_state'] = use_previous_state
                                            yield conf
                                    else:
                                        raise ValueError(f'{model_name} is not defined')


def get_data_config(name: str, single_spike: bool = False): # is one of dataset.DATA_NAMES
    if name in diffusion_functions.keys():
        h, w = 10, 7
        conf = {
            'num_nodes': h * w,
            'generator': lambda _: simple_grid_coo(h, w, directed=False),
            'num_initial_spikes': (h*w) // 3 if not single_spike else 1,
            't_max': 1000,
            'num_samples': 100,
            'min_sample_distance': 1,
            'diffusion_function': diffusion_functions[name],
            'heat_spike': (10., 15.),
            'cold_spike': (-15., -10.),
            'prob_cold_spike': 0.4 if not single_spike else 0,
            'step_size': 1e-3,
        }
    elif name in traffic_ablation_names:
        conf = {}
    else:
        conf = {
            'num_samples': lambda data_len: data_len // 3,
            'min_sample_distance': 1,
        }

    return conf


our = lambda input_dim, output_dim, time_dim: get_conf(input_dim, output_dim, time_dim)
c0 = lambda input_dim, output_dim, time_dim: get_conf_baseline('DCRNN', input_dim, output_dim, time_dim)
c1 = lambda input_dim, output_dim, time_dim: get_conf_baseline('GCRN_LSTM', input_dim, output_dim, time_dim)
c2 = lambda input_dim, output_dim, time_dim: get_conf_baseline('GCRN_GRU', input_dim, output_dim, time_dim)
c3 = lambda input_dim, output_dim, time_dim: get_conf_baseline('TGCN', input_dim, output_dim, time_dim)
c4 = lambda input_dim, output_dim, time_dim: get_conf_baseline('A3TGCN', input_dim, output_dim, time_dim)
c5 = lambda input_dim, output_dim, time_dim: get_conf_node('NODE', input_dim, output_dim, time_dim)
c6 = lambda input_dim, output_dim, time_dim: get_conf_node('NDCN', input_dim, output_dim, time_dim)

MODEL_CONF = {
    'TGODE': (our, TemporalGraphEuler), # Our method
    
    'DCRNN': (c0, DCRNNModel), 
    'GCRN_LSTM': (c1, GCRN_LSTM_Model), 
    'GCRN_GRU': (c2, GCRN_GRU_Model),
    'TGCN': (c3, TGCNModel), 
    'A3TGCN': (c4, A3TGCNModel),
    'NODE': (c5, NeuralODE),
    'NDCN': (c6, NDCN)
}
