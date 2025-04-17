import os
import torch
import numpy as np
from typing import Optional
from numpy.random import Generator, default_rng
from tsl.datasets.pems_benchmarks import PeMS04
from torch_geometric.data import Data, InMemoryDataset

traffic_ablation_names = [f'pems04_ablation_{n}' for n in [500 * (2**i) for i in range(6)]]

class TrafficAblationDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 name: str,
                 rng: Optional[Generator] = None) -> None:
        assert name in traffic_ablation_names
        self.name = name

        self.num_samples = int(name.split('_')[-1].strip())
        self.rng = rng if rng is not None else default_rng()

        super().__init__(root=root)
        self.data, self.slices, self.keeped_tmstp,\
            self.input_dim, self.output_dim, self.time_dim = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return [f'{self.name}.pt']

    def process(self):
        print('Builing the dataset...')

        dataset = PeMS04(self.root)
    
        edge_index, edge_attr = dataset.get_connectivity(threshold=0.1,
                                                        include_self=False,
                                                        normalize_axis=1,
                                                        layout="edge_index")
        df = dataset.dataframe()
        
        id_list = np.arange(len(df))
        self.rng.shuffle(id_list)

        to_keep = id_list[:self.num_samples]
        to_keep = sorted(to_keep)

        data_list = []
        for i in range(len(to_keep)-1):
            row = df.iloc[to_keep[i]]
            row_next = df.iloc[to_keep[i+1]]
            
            x = torch.tensor(row.to_numpy()).unsqueeze(1).float()
            y = torch.tensor(row_next.to_numpy()).unsqueeze(1).float()

            timestamp = df.index[to_keep[i]]
            
            # encoded_timestamp = torch.tensor([
            #     np.sin(2 * np.pi * timestamp.minute/60),
            #     np.cos(2 * np.pi * timestamp.minute/60),
            #     np.sin(2 * np.pi * timestamp.hour/24),
            #     np.cos(2 * np.pi * timestamp.hour/24),
            #     np.sin(2 * np.pi * timestamp.day/31),
            #     np.cos(2 * np.pi * timestamp.day/31),
            #     np.sin(2 * np.pi * timestamp.month/12),
            #     np.cos(2 * np.pi * timestamp.month/12),
            # ], dtype=torch.float)
            # encoded_timestamp = encoded_timestamp.repeat(x.shape[0],1)
            
            data_list.append(
                Data(
                    x = x,
                    y = y,
                    edge_index = torch.LongTensor(edge_index),
                    edge_attr = torch.FloatTensor(edge_attr),
                    t = timestamp,
                    #t_enc = encoded_timestamp,
                    delta_t = to_keep[i+1] - to_keep[i] #float(to_keep[i+1] - to_keep[i])
                )
            )

        input_dim = data_list[0].x.shape[-1]
        output_dim = data_list[0].y.shape[-1]
        time_dim = (data_list[0].t_enc.shape[-1] if hasattr(data_list[0], 't_enc') 
                    else None)

        data, slices = self.collate(data_list)
        torch.save(
            (data, slices, to_keep, input_dim, output_dim , time_dim), 
            self.processed_paths[0]
        )
        
        