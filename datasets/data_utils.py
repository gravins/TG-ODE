from numgraph.utils.spikes_generator import ColdHeatSpikeGenerator
import numpy as np
from numpy.random import default_rng


def sample_with_minimum_distance(n, k, d, rng):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """

    sample = list(range(n-(k-1)*(d-1)))
    rng.shuffle(sample)
    sample = sample[:k]

    def ranks(sample):
        """
        Return the ranks of each element in an integer sample.
        """
        indices = sorted(range(len(sample)), key=lambda i: sample[i])
        return sorted(indices, key=lambda i: indices[i])

    return sorted([s + (d-1)*r for s, r in zip(sample, ranks(sample))])


class MultiSpikeGenerator(ColdHeatSpikeGenerator):
    def __init__(self,
                 t_max = 10, 
                 num_initial_spikes = 1,
                 heat_spike = (15., 5.),
                 cold_spike = (-5., -15.),
                 prob_cold_spike = 0.4,
                 num_spikes = 1,
                 rng = None) -> None:
         
        self.num_initial_spikes = num_initial_spikes
        super().__init__(t_max, heat_spike ,cold_spike ,prob_cold_spike ,num_spikes ,rng)
        
    def compute_spike(self, t, x):
        if t in self.spike_timesteps:
            assert self.num_initial_spikes <= x.shape[0]
            nodes = list(range(x.shape[0]))

            self.rng.shuffle(nodes)
            for i in nodes[:self.num_initial_spikes]:
                # Improve heat of a random node
                if self.rng.uniform(low=0, high=1) < self.prob_cold_spike:
                    x[i,0] = self.rng.uniform(low=self.cold_spike[0], 
                                            high=self.cold_spike[1],
                                            size=(1, 1))
                else:
                    x[i,0] = self.rng.uniform(low=self.heat_spike[0], 
                                            high=self.heat_spike[1],
                                            size=(1, 1))
        return x
 

def heat_diffusion(edges, weights, num_nodes, x, f):
    # Compute the Laplacian matrix
    adj_mat = np.zeros((num_nodes, num_nodes))
    adj_mat[edges[:, 0], edges[:, 1]] = 1 if weights is None else weights
    np.fill_diagonal(adj_mat, 0)
    degree = np.diag(np.sum(adj_mat, axis=1))
    new_degree = np.linalg.inv(np.sqrt(degree))
    L = np.eye(num_nodes) - new_degree @ adj_mat @ new_degree # Normalized laplacian
    return - f(L) @ x


diffusion_functions = {
    'heat': lambda edges, weights, num_nodes, x: heat_diffusion(edges, weights, num_nodes, x, f=lambda L: L),
    'pow_2_heat': lambda edges, weights, num_nodes, x: heat_diffusion(edges, weights, num_nodes, x, f=lambda L: np.linalg.matrix_power(L, 2)),
    'pow_5_heat': lambda edges, weights, num_nodes, x: heat_diffusion(edges, weights, num_nodes, x, f=lambda L: np.linalg.matrix_power(L, 5)),
    'tanh_heat': lambda edges, weights, num_nodes, x: heat_diffusion(edges, weights, num_nodes, x, f=lambda L: np.tanh(L)),
    'expand_heat': lambda edges, weights, num_nodes, x: heat_diffusion(edges, weights, num_nodes, x, f=lambda L: 5*L),
    'reduce_heat': lambda edges, weights, num_nodes, x: heat_diffusion(edges, weights, num_nodes, x, f=lambda L: 0.05*L),
    'gaussian_noise_heat': lambda edges, weights, num_nodes, x: heat_diffusion(edges, weights, num_nodes, x, f=lambda L: (default_rng(9).normal(size=L.shape) * (L!=0)) + L)
}
