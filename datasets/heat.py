import os
import tqdm
import torch

from typing import Optional, Callable, Tuple
from numpy.random import Generator, default_rng
from torch_geometric.data import Data, InMemoryDataset
from numgraph.temporal import euler_graph_diffusion_coo 
from .data_utils import sample_with_minimum_distance, MultiSpikeGenerator, diffusion_functions


class MultiSpikeHeatDataset(InMemoryDataset):
    """
    Spatio-Temporal graph simulating the heat diffusion over a graph.
    Initial condition: all nodes have a temperature between 0. and 0.2,
    exept for one node that is hotter. Only one spike is considered.
    """
    def __init__(self, 
                 root: str,
                 name: str,
                 num_nodes: int,
                 generator: Callable,
                 num_initial_spikes: int = None,
                 t_max: int = 1000,
                 num_samples: int = 100,
                 min_sample_distance: int = 1,
                 diffusion_function: Optional[Callable] = None, 
                 heat_spike: Tuple[float, float] = (0.7, 2.),
                 cold_spike: Tuple[float, float] = (-2., -0.7),
                 prob_cold_spike = 0.4,
                 step_size: float = 0.1,
                 name_suffix: str = '', # train, valid, test
                 rng: Optional[Generator] = None):

        self.root = root
        self.name = name
        self.suffix = name_suffix
        self.num_nodes = num_nodes
        self.generator = generator

        assert num_initial_spikes is None or num_initial_spikes <= num_nodes, f"num_initial_spikes should be smaller or equal to num_nodes; got {num_initial_spikes} and {num_nodes}"
        self.num_initial_spikes = (num_initial_spikes if num_initial_spikes is not None else 
                                   max(1, num_nodes // 3))

        self.t_max = t_max
        assert num_samples * min_sample_distance <= t_max, f"num_samples * min_sample_distance must be <= than t_max; got {num_samples}, {min_sample_distance}, and {t_max}"
        self.num_samples = num_samples
        self.min_sample_distance = min_sample_distance
        self.num_spikes = 1

        self.diffusion = diffusion_function
        self.heat_spike = heat_spike
        self.cold_spike = cold_spike
        self.prob_cold_spike = prob_cold_spike
        
        self.step_size = step_size
        self.rng = rng if rng is not None else default_rng()

        super().__init__(root)
        self.data, self.slices, self.keeped_tmstp, self.spikegen = torch.load(self.processed_paths[0])
        self.input_dim = self[0].x.shape[-1]
        self.output_dim = self[0].y.shape[-1]
        self.time_dim = None

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        name = f'{self.name}_{self.suffix}.pt'
        return [name]

    def process(self):
        print('Builing the dataset...')
        spikegen = MultiSpikeGenerator(
            num_initial_spikes = self.num_initial_spikes,
            cold_spike = self.cold_spike,
            prob_cold_spike = self.prob_cold_spike,
            t_max = self.t_max,
            heat_spike = self.heat_spike,
            num_spikes = self.num_spikes,
            rng = self.rng
        )

        # if diffusion is None then -Lx is used
        snapshots, xs = euler_graph_diffusion_coo(
            self.generator, spikegen, diffusion=self.diffusion,
            step_size=self.step_size, t_max=self.t_max, num_nodes=self.num_nodes
        )

        edges, weights = snapshots[0]
        edge_index = torch.from_numpy(edges.T)
    
        to_keep = sample_with_minimum_distance(
            n = self.t_max, 
            k = self.num_samples, 
            d = self.min_sample_distance, 
            rng = self.rng
        )
        
        data = []
        prev_x = torch.from_numpy(xs[0]).float()
        prev_i = 0
        for i in tqdm.tqdm(to_keep):
            y = torch.from_numpy(xs[i]).float()

            data.append(
                Data(
                    edge_index = edge_index, 
                    x = prev_x, # initial condition for diffusion, ie, x[i-1]
                    y = y,  # the temperature of the nodes at time i
                    delta_t = i - prev_i, #float(i - prev_i),
                    intermediate = torch.stack([torch.from_numpy(xs[j]).float() for j in range(prev_i, i+1)])
                )
            )
            prev_x = y
            prev_i = i
            
        data, slices = self.collate(data)
        torch.save((data, slices, to_keep, spikegen), self.processed_paths[0])