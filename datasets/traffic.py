import os
import torch
import numpy as np
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from torch_geometric.data import Data, InMemoryDataset
from numpy.random import Generator, default_rng
from typing import Optional, Union, Callable
from .data_utils import sample_with_minimum_distance
from torch_geometric_temporal.dataset import MontevideoBusDatasetLoader


traffic_data_name = ['metrla', 'pems03', 'pems04', 'pems07', 'pems08', 'pemsbay', 'montevideo']

class TrafficForecastingDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 name: str,
                 num_samples: Optional[Union[Callable, int]] = None, # it can be a number or a function that computes the number of nodes given len(data_list). By default, it is len(data_list) // self.min_sample_distance
                 min_sample_distance: int = 1,
                 rng: Optional[Generator] = None) -> None:
        self.name = name
        
        self.num_samples = num_samples
        self.min_sample_distance = min_sample_distance
        assert self.name in traffic_data_name, 'Dataset name can be only one of {traffic_data_name}'
        
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

        if self.name == 'montevideo':
            data_list, to_keep = self.process_montevideo()
        else:
            if self.name == 'metrla':
                dataset = MetrLA(self.root, impute_zeros=True)
            elif self.name == 'pemsbay':
                dataset = PemsBay(self.root)
            elif self.name == 'pems03':
                dataset = PeMS03(self.root)
            elif self.name == 'pems04':
                dataset = PeMS04(self.root)
            elif self.name == 'pems07':
                dataset = PeMS07(self.root)
            elif self.name == 'pems08':
                dataset = PeMS08(self.root)
            else:
                raise NotImplementedError()

            edge_index, edge_attr = dataset.get_connectivity(threshold=0.1,
                                                            include_self=False,
                                                            normalize_axis=1,
                                                            layout="edge_index")
            df = dataset.dataframe()
            
            # Computes the number of num samples
            if self.num_samples is None:
                tmp = len(df) // self.min_sample_distance
            elif isinstance(self.num_samples, int):
                tmp = self.num_samples
            else:
                tmp = self.num_samples(len(df))

            to_keep = sample_with_minimum_distance(
                n = len(df), 
                k = tmp,
                d = self.min_sample_distance,
                rng = self.rng
            )
            data_list = []
            for i in range(len(to_keep)-1):
                row = df.iloc[to_keep[i]]
                row_next = df.iloc[to_keep[i+1]]
                
                x = torch.tensor(row.to_numpy()).unsqueeze(1).float()
                y = torch.tensor(row_next.to_numpy()).unsqueeze(1).float()

                timestamp = df.index[to_keep[i]]
                
                encoded_timestamp = torch.tensor([
                    np.sin(2 * np.pi * timestamp.minute/60),
                    np.cos(2 * np.pi * timestamp.minute/60),
                    np.sin(2 * np.pi * timestamp.hour/24),
                    np.cos(2 * np.pi * timestamp.hour/24),
                    np.sin(2 * np.pi * timestamp.day/31),
                    np.cos(2 * np.pi * timestamp.day/31),
                    np.sin(2 * np.pi * timestamp.month/12),
                    np.cos(2 * np.pi * timestamp.month/12),
                ], dtype=torch.float)
                encoded_timestamp = encoded_timestamp.repeat(x.shape[0],1)
                
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
        
    def process_montevideo(self):
        dataset = MontevideoBusDatasetLoader()
        dataset = dataset.get_dataset(lags = 1)

        # Computes the number of num samples
        if self.num_samples is None:
            tmp = len(dataset) // self.min_sample_distance
        elif isinstance(self.num_samples, int):
            tmp = self.num_samples
        else:
            tmp = self.num_samples(len(dataset.features))

        to_keep = sample_with_minimum_distance(
            n = len(dataset.features), 
            k = tmp,
            d = self.min_sample_distance,
            rng = self.rng
        )

        data_list = []
        for i in range(len(to_keep)-1):
            data = dataset[to_keep[i]]
            data.y = data.y.unsqueeze(dim=1)
            setattr(data, 't', to_keep[i])
            setattr(data, 'delta_t', to_keep[i+1] - to_keep[i])
            data_list.append(data)

        return data_list, to_keep