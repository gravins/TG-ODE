import os
import torch
import numpy as np
from tsl.data.preprocessing.scalers import StandardScaler, MinMaxScaler
from datasets import (MultiSpikeHeatDataset, TrafficForecastingDataset, 
                      traffic_data_name, traffic_ablation_names, 
                      TrafficAblationDataset)


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true)) 


def get_dataset(root, name, args, ts_size=0., vl_size=0., x_scaler='', t_scaler='', device='cpu'):
    if (ts_size > 0 or vl_size > 0) and name not in traffic_data_name:
        print(f'Got ts_size={ts_size} vl_size={vl_size}, and dataset_name={name}. Cannot use ts_size and/or vl_size > 0 with {name} diffusion. Thus, ts_size and vl_size are NOT going to be used.')

    x_scaler = SCALERS[x_scaler]((0,1)) if SCALERS[x_scaler] is not None else None
    t_scaler = SCALERS[t_scaler](0) if SCALERS[t_scaler] is not None else None

    if name in traffic_data_name or name in traffic_ablation_names:
        if name in traffic_data_name:
            dataset = TrafficForecastingDataset(root=root, name=name, **args)
        else:
            dataset = TrafficAblationDataset(root=root, name=name, **args)
        input_dim = dataset.input_dim
        output_dim = dataset.output_dim
        time_dim = dataset.time_dim

        ids = np.arange(len(dataset))
        # TR-TS split
        tmp = int(len(ids) * ts_size)
        ts_ids = ids[-tmp:]
        tr_ids = ids[:-tmp]
        # TR-VL split
        tmp = int(len(tr_ids) * vl_size)
        vl_ids = tr_ids[-tmp:]
        tr_ids = tr_ids[:-tmp]

        if x_scaler is not None:
            x, t = [], []
            for d in dataset[tr_ids]:
                x.append(d.x)
                if t_scaler is not None: t.append(d.delta_t)
            x = torch.stack(x)
            x_scaler.fit(x)
            if t_scaler is not None: 
                t = torch.stack(t, axis=-1).squeeze()
                t_scaler.fit(t)

        train_dataset = dataset[tr_ids]
        valid_dataset = dataset[vl_ids]
        test_dataset  = dataset[ts_ids]     
    else:
        ## Be careful by tuning this parameters! You can end up in a situation where you reach the 
        ## convergence state in a few steps. After convergence, the nodes' values are always near the avg.
        args['name_suffix'] = 'train'
        train_dataset = MultiSpikeHeatDataset(root=root, name=name, **args)
        input_dim = train_dataset.input_dim
        output_dim = train_dataset.output_dim
        time_dim = train_dataset.time_dim
        args['name_suffix'] = 'valid'
        args['t_max'] //= 2
        args['num_samples'] //= 2
        valid_dataset = MultiSpikeHeatDataset(root=root, name=name, **args)
        args['name_suffix'] = 'test'
        test_dataset = MultiSpikeHeatDataset(root=root, name=name, **args)

        if x_scaler is not None:
            x, t = [], []
            for d in train_dataset:
                x.append(d.x)
                if t_scaler is not None: t.append(d.delta_t)
            x = torch.stack(x)
            x_scaler.fit(x)
            if t_scaler is not None: 
                t = torch.stack(t, axis=-1).squeeze()
                t_scaler.fit(t)
    
    train_dataset.data.to(device)
    valid_dataset.data.to(device)
    test_dataset.data.to(device)
    if x_scaler is not None:
        x_scaler.scale = x_scaler.scale.to(device)
        x_scaler.bias = x_scaler.bias.to(device)
    if t_scaler is not None:
        t_scaler.scale = t_scaler.scale.to(device)
        t_scaler.bias = t_scaler.bias.to(device)

    return train_dataset, valid_dataset, test_dataset, input_dim, output_dim, time_dim, x_scaler, t_scaler


def compute_scores(y_pred, y_true):
    return {k: func(y_pred, y_true).detach().cpu().item() for k, func in SCORES.items()}


SCALERS = {
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    '': None,
    None: None
}

SCORES = {
    'MAE': torch.nn.L1Loss(),
    'MSE': torch.nn.MSELoss(),
    'RMSE': RMSELoss()
}