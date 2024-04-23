from train_eval import train_and_eval, train_and_eval_single
from util import create_if_not_exists, get_dataset
from conf import MODEL_CONF
import pandas as pd
import numpy as np
import tqdm
import math
import ray
import os


def update_results(pbar, res, df, best_score, criterion_name, model_dir, file_name='model_selection_partial.csv'):
    # Write conf id
    if 'conf_name' not in df:
        df['conf_name'] = [res['opt']['conf_name']]
    else:
        df['conf_name'].append(res['opt']['conf_name'])

    # Write results
    for k in res['best'].keys():
        if 'other' in k:
            for metric in res['best'][k]:
                kk = k.replace('scores', metric) # eg, tr_other_MSE
                if kk not in df:
                    df[kk] = [res['best'][k][metric]]
                else:
                    df[kk].append(res['best'][k][metric])
        else:
            kk = k.replace('loss', res['opt']['exp']['criterion']) # eg, tr_MAE
            if kk not in df:
                df[kk] = [res['best'][k]]
            else:
                df[kk].append(res['best'][k])

    # Write experimental parameters
    for prefix in ['model_params', 'optim_params', 'data', 'exp']:
        for k in res['opt'][prefix].keys():
            if f'{prefix.replace("_params", "")}_{k}' not in df:
                df[f'{prefix.replace("_params", "")}_{k}'] = [res['opt'][prefix][k]]
            else:
                df[f'{prefix.replace("_params", "")}_{k}'].append(res['opt'][prefix][k])

    # Update the progress bar
    if best_score is None or res['best']['vl_loss'] < best_score:
        pbar.set_postfix(
            best_vl_loss = res['best']['vl_loss'],
            best_ts_loss = res['best']['ts_loss'],
            best_tr_loss = res['best']['tr_loss'],
            #other_vl_scores = res['best']['vl_other_scores'],
            #other_ts_scores = res['best']['ts_other_scores'],
        )    
        best_score = res['best']['vl_loss']
    pbar.update(1)

    df_ = pd.DataFrame(df)
    df_ = df_.sort_values(f'vl_{criterion_name}', ascending=True)
    df_.to_csv(os.path.join(model_dir, file_name), index=False)

    return pbar, df, best_score


def write_ntrial_res(df, criterion_name, model_dir, is_torchdyn_used):
        ts_scores = df[f'ts_{criterion_name}'].values
        log_ts_scores = [math.log10(s) for s in ts_scores]
        mean, std = np.mean(ts_scores), np.std(ts_scores)
        log_mean, log_std = np.mean(log_ts_scores), np.std(log_ts_scores)
        with open(os.path.join(model_dir,'final_res.txt'), 'a') as f:
            f.write(f'torchdyn={is_torchdyn_used} score: ' + str(mean) + '$_{\pm' + str(std) + '}$\n\t' + f'torchdyn={is_torchdyn_used} score: ' + str(round(mean,3)) + '$_{\pm' + str(round(std,3)) + '}$\n')
            f.flush()
            f.write(f'torchdyn={is_torchdyn_used} log score: ' + str(log_mean) + '$_{\pm' + str(log_std) + '}$\n\t' + f'torchdyn={is_torchdyn_used} log score: ' + str(round(log_mean,3)) + '$_{\pm' + str(round(log_std,3)) + '}$\n')
            f.flush()
            f.close()


def select_and_assess(seed: int,
                      ntrials: int,
                      model_name: str, # the model name (one of MODEL_CONF[model_name])
                      data_name: str, # the dataset name (one of datasets.DATA_NAMES)
                      n_epochs: int,
                      patience: int,
                      batch_size: int,
                      criterion_name: str,
                      data_params: dict,
                      exp_dir: str, # ./RESULTS/
                      vl_size: float,
                      ts_size: float,
                      x_scaler: str = '',
                      t_scaler: str = '',
                      debug: bool = False): 
                      #device: str = 'cpu'): 

    model_dir = os.path.join(exp_dir, data_name, model_name) # ./RESULTS/metrla/GCLSTM
    create_if_not_exists(model_dir)

    ckpt_dir = os.path.join(model_dir, 'checkpoints') # ./RESULTS/metrla/GCLSTM/checkpoints
    create_if_not_exists(ckpt_dir)

    # Generate datasets
    _ ,_ ,_, input_dim, output_dim, time_dim, _, _ = get_dataset( # dataset is saved at ./RESULTS/metrla/processed
        root = exp_dir,
        name = data_name,
        args = data_params,
        vl_size = vl_size,
        ts_size = ts_size,
        x_scaler = x_scaler,
        t_scaler = t_scaler
    )

    #results_dir = os.path.join(model_dir, 'results')
    #create_if_not_exists(results_dir)

    get_conf, model_instance = MODEL_CONF[model_name]
    df = {}
    ray_ids = []
    best_score = None
    pbar = tqdm.tqdm(total=len([1 for _ in get_conf(input_dim, output_dim, time_dim)]))
    if os.environ.get("CUDA_VISIBLE_DEVICES", '') != '':
        max_concurrency = int(len(os.environ.get("CUDA_VISIBLE_DEVICES").split(',')) / float(os.environ.get("PERC_GPUS")))
    else:
        max_concurrency = int(os.environ.get("NUM_CPUS", 1))

    for i, conf in enumerate(get_conf(input_dim, output_dim, time_dim)):
        model_params = conf['model']
        optim_params = conf['optim']
        opt = {
            'data': data_params,
            'model_params': model_params,
            'optim_params': optim_params,
            'exp':{
                'seed': seed,
                'root': exp_dir, 
                'data_name': data_name,
                'ckpt_dir': ckpt_dir,
                'epochs': n_epochs,
                'criterion': criterion_name,
                'exp_name': f'{model_name}_{data_name}',
                'patience': patience,
                'batch_size': batch_size,
                'vl_size': vl_size,
                'ts_size': ts_size,
                'x_scaler': x_scaler,
                't_scaler': t_scaler
            },
            'conf_name': f'conf_{i}_{seed}'
        }

        if debug:
            train_and_eval_single(model_instance, opt)
        else:
            ray_ids.append(train_and_eval.remote(model_instance, opt))
            while len(ray_ids) >= max_concurrency:
                done_id, ray_ids = ray.wait(ray_ids)
                res = ray.get(done_id[0])
                pbar, df, best_score = update_results(pbar, res, df, best_score, criterion_name, model_dir)

    if debug: exit()
    
    while len(ray_ids):
        done_id, ray_ids = ray.wait(ray_ids)
        res = ray.get(done_id[0])
        pbar, df, best_score = update_results(pbar, res, df, best_score, criterion_name, model_dir)
    
    # Save model selection resutls
    df = pd.DataFrame(df)
    df = df.sort_values(f'vl_{criterion_name}', ascending=True)
    df.to_csv(os.path.join(model_dir, 'model_selection.csv'), index=False)

    # Run the best config multiple times
    exp_seeds = set([seed])
    exp_seeds.update(list(range(ntrials-1)))
    if len(exp_seeds) < ntrials:
        exp_seeds.append([ntrials])
    exp_seeds.remove(seed)


    confs = [conf for conf in get_conf(input_dim, output_dim, time_dim)]
    best_confs = {}
    if 'torchdyn_method' in confs[0]['model']:
        for v, gdf in df.groupby('model_torchdyn_method'):
            gdf = gdf.sort_values(f'vl_{criterion_name}', ascending=True)
            best_confs[f'torchdyn={v}'] = gdf.iloc[0]
    else:
        best_confs['torchdyn=False'] = df.iloc[0]

    df = {}
    ray_ids = []
    best_score = None
    pbar = tqdm.tqdm(total=len(best_confs)*(ntrials-1))
    for is_torchdyn_used, df_best_row in best_confs.items():
        best_id = int(df_best_row['conf_name'].split("_")[1])
        conf = confs[best_id]

        for seed in exp_seeds:
            model_params = conf['model']
            optim_params = conf['optim']
            opt = {
                'data': data_params,
                'model_params': model_params,
                'optim_params': optim_params,
                'exp':{
                    'seed': seed,
                    'root': exp_dir, 
                    'data_name': data_name,
                    'ckpt_dir': ckpt_dir,
                    'epochs': n_epochs,
                    'criterion': criterion_name,
                    'exp_name': f'{model_name}_{data_name}',
                    'patience': patience,
                    'batch_size': batch_size,
                    'vl_size': vl_size,
                    'ts_size': ts_size,
                    'x_scaler': x_scaler,
                    't_scaler': t_scaler
                },
                'conf_name': f'conf_{best_id}_{seed}'
            }

            if debug:
                train_and_eval_single(model_instance, opt)
            else:
                ray_ids.append(train_and_eval.remote(model_instance, opt))
                while len(ray_ids) >= max_concurrency:
                    done_id, ray_ids = ray.wait(ray_ids)
                    res = ray.get(done_id[0])
                    pbar, df, best_score = update_results(pbar, res, df, best_score, criterion_name, model_dir, file_name=f'best_conf_assessment_{ntrials}_ntrials_partial.csv')

    while len(ray_ids):
        done_id, ray_ids = ray.wait(ray_ids)
        res = ray.get(done_id[0])
        pbar, df, best_score = update_results(pbar, res, df, best_score, criterion_name, model_dir, file_name=f'best_conf_assessment_{ntrials}_ntrials_partial.csv')

    # Save best conf results
    df = pd.DataFrame(df)
    for is_torchdyn_used, df_best_row in best_confs.items():
        df = df.append(df_best_row)
    df.to_csv(os.path.join(model_dir, f'best_conf_assessment_{ntrials}_ntrials.csv'), index=False)

    if len(best_confs) > 1:
        for is_torchdyn_used, gdf in df.groupby('model_torchdyn_method'):
            write_ntrial_res(gdf, criterion_name, model_dir, is_torchdyn_used)
    else:
        write_ntrial_res(df, criterion_name, model_dir, False)