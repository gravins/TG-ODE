# TG-ODE
This repository provides the official reference implementation of our paper **_"Temporal Graph ODEs for Irregularly-Sampled Time Series"_** accepted at the International Joint Conference on Artificial  Intelligence (IJCAI) 2024.

Please consider citing us

	@inproceedings{gravina2024tgode,
	  title     = {Temporal Graph ODEs for Irregularly-Sampled Time Series},
	  author    = {Gravina, Alessio and Zambon, Daniele and Bacciu, Davide and Alippi, Cesare},
	  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
	  publisher = {International Joint Conferences on Artificial Intelligence Organization},
	  editor    = {Kate Larson},
	  pages     = {4025--4034},
	  year      = {2024},
	  month     = {8},
	  note      = {Main Track},
	  doi       = {10.24963/ijcai.2024/445},
	  url       = {https://doi.org/10.24963/ijcai.2024/445},
	}




## Requirements
_Note: we assume Miniconda/Anaconda is installed, otherwise see this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for correct installation. The proper Python version is installed during the first step of the following procedure._

1. Install the required packages and create the environment
    - ``` conda env create -f env.yml ```

2. Activate the environment
    - ``` conda activate tgode ```


## How to reproduce our experiments
First, extract the preprocessed data through the command: ```tar -xvf RESULTS.tar.xz```


Then:

```
export data="" # choose one from ['metrla', 'pems03', 'pems04', 'pems07', 'pems08', 'montevideo', 'heat', 'pow_2_heat', 'pow_5_heat', 'tanh_heat', 'expand_heat', 'reduce_heat', 'gaussian_noise_heat']
export model="" # choose one from ['DCRNN', 'GCRN_LSTM', 'GCRN_GRU', 'TGCN', 'A3TGCN', 'NODE', 'GDE', 'TGODE']
export NUM_CPUS=90 # number of available cpus for the entire experiment
export PERC_GPUS=0.0 # percentage of gpus for one configuration
export CUDA_VISIBLE_DEVICES="" # list of cuda visible devices
```

- Single-spike heat diffusion
```
export batch=16
export dir=RESULTS/single_spike/$data/
nohup python3 -u main.py --singlespike --data $data --model $model --batch $batch --savedir $dir --x_scaler StandardScaler >$dir/out_$model_$data 2>$dir/err_$model_$data
```

- Multi-spike heat diffusion
```
export batch=16
export dir=RESULTS/multi_spike/$data/
nohup python3 -u main.py --data $data --model $model --batch $batch --savedir $dir --x_scaler StandardScaler >$dir/out_$model_$data 2>$dir/err_$model_$data
```

- Traffic forecasting
```
export batch=1
export dir=RESULTS/$data/
nohup python3 -u main.py --data $data --model $model --batch $batch --savedir $dir --x_scaler StandardScaler >$dir/out_$model_$data 2>$dir/err_$model_$data
```
