from .traffic import TrafficForecastingDataset, traffic_data_name
from .heat import MultiSpikeHeatDataset, diffusion_functions
from .traffic_ablation import TrafficAblationDataset, traffic_ablation_names


DATA_NAMES = traffic_data_name + list(diffusion_functions.keys()) + traffic_ablation_names
