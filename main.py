import os
import torch

import ray
import time
import random
import datetime
import argparse
import numpy as np
from datasets import DATA_NAMES
from util import SCORES, SCALERS
from numpy.random import default_rng
from conf import MODEL_CONF, get_data_config
from model_selection_and_assessment import select_and_assess

print('\tOMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS'))
print(f'\tUsable GPUs: {os.environ.get("CUDA_VISIBLE_DEVICES")}, percent of GPUs per worker: {os.environ.get("PERC_GPUS")}')
print(f'\tUsable CPUs (virual cores): {os.environ.get("NUM_CPUS")}')


if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', 
                        help='The name of the dataset to load.',
                        default=DATA_NAMES[0],
                        choices=DATA_NAMES)
    parser.add_argument('--singlespike', 
                        help='If you want only one initial spike in the heat diffusion.',
                        action='store_true')
    parser.add_argument('--model',
                        help='The model name.',
                        default=list(MODEL_CONF)[0],
                        choices=MODEL_CONF)
    parser.add_argument('--epochs', help='The number of epochs.', default=3000, type=int)
    parser.add_argument('--batch', help='The batch size.', default=128, type=int)
    parser.add_argument('--patience', 
                        help='Training stops if the selected metric does not improve for X epochs',
                        type=int,
                        default=100)
    parser.add_argument('--ntrials',
                        help='The number of trials for the best config.', 
                        default=5,
                        type = float)
    parser.add_argument('--savedir', help='The saving directory.', default='.')
    parser.add_argument('--metric',
                        help='The matric that will be optimized.', 
                        default='MAE',
                        choices= SCORES,
                        type = str)
    parser.add_argument('--vl_size',
                        help='The size of the validation set in percentage.', 
                        default=0.1,
                        type = float)
    parser.add_argument('--ts_size',
                        help='The size of the test set in percentage.', 
                        default=0.1,
                        type = float)
    parser.add_argument('--x_scaler',
                        help='The scaler for the data x.', 
                        default='',
                        choices= SCALERS,
                        type = str)
    parser.add_argument('--t_scaler',
                        help='The scalers for the data delta_t.', 
                        default='',
                        choices= SCALERS,
                        type = str)
    parser.add_argument('--seed', help='The seed of the experiment.', default=12345, type = int)
    parser.add_argument('--debug', help='Debug mode.', action='store_true')
    parser.add_argument('--cluster', help='Slurm cluster mode.', action='store_true')
    args = parser.parse_args()

    assert args.ts_size < 1 and args.ts_size > 0, 'Test size must be in (0,1), got {args.ts_size}'
    assert args.vl_size < 1 and args.vl_size > 0, 'Validation size must be in (0,1), got {args.vl_size}'

    print(f'Running experiments with the followeing args {args}')

    if not args.debug:
        if args.cluster: 
            ray.init(address=os.environ.get("ip_head"), _redis_password=os.environ.get("redis_password"))
        else:
            num_gpus = os.environ.get("CUDA_VISIBLE_DEVICES").split(',')
            num_gpus = 0 if len(num_gpus) == 1 and num_gpus[0] == '' else len(num_gpus)
            ray.init(num_cpus=int(os.environ.get("NUM_CPUS")),
                    num_gpus=num_gpus) # local ray initialization

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = default_rng(args.seed)

    data_conf = get_data_config(args.data, args.singlespike)
    data_conf['rng'] = rng

    select_and_assess(
        seed = args.seed,
        ntrials = args.ntrials,
        model_name = args.model,
        data_name = args.data,
        n_epochs = args.epochs,
        patience = args.patience,
        batch_size = args.batch,
        criterion_name = args.metric,
        data_params = data_conf,
        x_scaler = args.x_scaler,
        t_scaler = args.t_scaler,
        vl_size = args.vl_size,
        ts_size = args.ts_size,
        exp_dir = args.savedir, # /gravina/idsia/
        debug = args.debug
    )
    
    elapsed = time.time() - t0
    print(datetime.timedelta(seconds=elapsed))
