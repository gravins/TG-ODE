import os
import torch
from util import get_dataset, RMSELoss, compute_scores
from torch.utils.data import DataLoader
import numpy as np
import pickle
import tqdm
import ray
import random

def optimizer_to(optim, device):
    # Code from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def train(model, tr_loader, optimizer, criterion, device, x_scaler=None, t_scaler=None):
    model.train()

    prev_h = None
    for batch in tr_loader:
        # Reset gradients from previous step
        model.zero_grad()

        y_pred, y_true = [], []
        for snapshot in batch:
            if x_scaler is not None: snapshot.x = x_scaler.transform(snapshot.x).squeeze(0)
            if t_scaler is not None: snapshot.delta_t = t_scaler.transform(snapshot.delta_t)
            #snapshot = snapshot.to(device)        
            
            # Perform a forward pass
            y, h = model.forward(snapshot, prev_h)
            prev_h = h

            y_pred.append(y.cpu())
            y_true.append(snapshot.y.cpu())

        # Perform a backward pass to calculate the gradients
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        loss = criterion(y_pred, y_true)
        loss.backward()

        # Update parameters
        optimizer.step()

        # if you don't detatch previous state you will get backprop error
        if prev_h is not None:
            prev_h = prev_h.detach()


def eval(model, loader, criterion, device, x_scaler=None, t_scaler=None):
    y_true, y_pred = [], []
    with torch.no_grad():
        prev_h = None
        for batch in loader:
            # Reset gradients from previous step
            model.zero_grad()

            for snapshot in batch:
                if x_scaler is not None: snapshot.x = x_scaler.transform(snapshot.x).squeeze(0)
                if t_scaler is not None: snapshot.delta_t = t_scaler.transform(snapshot.delta_t)
                #snapshot = snapshot.to(device)

                # Perform a forward pass
                y, h = model.forward(snapshot, prev_h)
                prev_h = h

                y_pred.append(y.cpu().detach())
                y_true.append(snapshot.y.cpu().detach())

    # Perform a backward pass to calculate the gradients
    loss = criterion(torch.cat(y_pred),
                     torch.cat(y_true))
 
    return loss, y_true, y_pred


@ray.remote(num_cpus=1, num_gpus=float(os.environ.get('PERC_GPUS', 0.)))
def train_and_eval(model_instance, opt):
    return train_and_eval_single(model_instance, opt)


def train_and_eval_single(model_instance, opt):
    print(f'Running {opt}...')

    # Set random seed
    random.seed(opt['exp']['seed'])
    np.random.seed(opt['exp']['seed'])
    torch.manual_seed(opt['exp']['seed'])
    torch.cuda.manual_seed_all(opt['exp']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_instance(**opt['model_params']).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt['optim_params']['lr'],
                                 weight_decay=opt['optim_params']['weight_decay'])
    if opt['exp']['criterion'] == 'MAE':
        criterion = torch.nn.L1Loss()
    elif opt['exp']['criterion'] == 'MSE':
        criterion = torch.nn.MSELoss()
    elif opt['exp']['criterion'] == 'RMSE':
        criterion = RMSELoss()
    else:
        raise NotImplementedError()

    best_score = (np.inf, np.inf, np.inf) # (tr_loss, vl_loss, ts_loss)
    best_epoch = 0
    history = []
    epochs = opt['exp']['epochs']
    path_save_best = os.path.join(opt["exp"]["ckpt_dir"], f'{opt["exp"]["exp_name"]}_{opt["conf_name"]}.pt')
    
    # LOAD previuos ckpt if exists
    if os.path.exists(path_save_best):
        # Load the existing checkpoint
        print(f'Loading {path_save_best}')
        ckpt = torch.load(path_save_best, map_location=device)
        best_epoch = ckpt['epoch']
        best_score = ckpt['best_score']
        history = ckpt['history']

        if 'train_ended' in ckpt and ckpt['train_ended']:
            print(f'{opt["model_params"]} has been already computed. I am not overriding it.')
            return {'opt': opt, 'best': history[best_epoch]}

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_to(optimizer, device) # Map the optimizer to the current device

    # Load data
    train_dataset, valid_dataset, test_dataset, _, _, _, x_scaler, t_scaler = get_dataset(
        root = opt['exp']['root'],
        name = opt['exp']['data_name'],
        args = opt['data'],
        ts_size = opt['exp']['ts_size'],
        vl_size = opt['exp']['vl_size'],
        x_scaler = opt['exp']['x_scaler'],
        t_scaler = opt['exp']['t_scaler'],
        device = device
    )

    collate_fn = lambda samples_list: samples_list
    tr_loader = DataLoader(train_dataset, batch_size=opt['exp']['batch_size'], collate_fn=collate_fn, shuffle=False) # 10 timesteps at a time
    vl_loader = DataLoader(valid_dataset, batch_size=opt['exp']['batch_size'], collate_fn=collate_fn, shuffle=False)
    ts_loader = DataLoader(test_dataset, batch_size=opt['exp']['batch_size'], collate_fn=collate_fn, shuffle=False)

    # RUN experiment
    for epoch in range(best_epoch, epochs): #tqdm.tqdm(range(opt['exp']['epochs'])):
        #print(f'Epoch {epoch}:')
        train(model, tr_loader, optimizer, criterion, device, x_scaler, t_scaler)
        
        # Check the scores 
        tr_loss, tr_y_true, tr_y_pred = eval(model, tr_loader, criterion, device, x_scaler, t_scaler) 
        vl_loss, vl_y_true, vl_y_pred = eval(model, vl_loader, criterion, device, x_scaler, t_scaler)
        ts_loss, ts_y_true, ts_y_pred = eval(model, ts_loader, criterion, device, x_scaler, t_scaler)

        tr_other_scores = compute_scores(torch.cat(tr_y_pred), torch.cat(tr_y_true))
        vl_other_scores = compute_scores(torch.cat(vl_y_pred), torch.cat(vl_y_true))
        ts_other_scores = compute_scores(torch.cat(ts_y_pred), torch.cat(ts_y_true))

        history.append({
            'tr_loss': tr_loss.detach().cpu().item(),
            'vl_loss': vl_loss.detach().cpu().item(),
            'ts_loss': ts_loss.detach().cpu().item(),
            'tr_other_scores': tr_other_scores,
            'vl_other_scores': vl_other_scores,
            'ts_other_scores': ts_other_scores,
        })

        if vl_loss < best_score[1] or best_score[1] == np.inf:
            best_ts_y_true, best_ts_y_pred = ts_y_true, ts_y_pred
            
            best_score = (tr_loss, vl_loss, ts_loss)
            best_epoch = epoch
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'loss': best_score,
                'tr_other_scores': tr_other_scores,
                'vl_other_scores': vl_other_scores,
                'ts_other_scores': ts_other_scores,
                'best_ts_y_true': best_ts_y_true, 
                'best_ts_y_pred': best_ts_y_pred,
                'history': history,
                'train_ended': False
            }, path_save_best)
        
        if epoch - best_epoch > opt['exp']['patience']:
            break


    ckpt = torch.load(path_save_best)
    ckpt['train_ended'] = True
    torch.save(ckpt, path_save_best)

    print(f'{opt["model_params"]}, {opt["optim_params"]}: Ended [{best_epoch}] Train {opt["exp"]["criterion"]}: {best_score[0]}, Val {opt["exp"]["criterion"]}: {best_score[1]}, Test {opt["exp"]["criterion"]}: {best_score[2]}, other Test {history[best_epoch]["ts_other_scores"]}')
    
    #best_res = {
    #    'losses': best_score,
    #    'epoch': best_epoch,
    #    'history': history,
    #    'exp_conf': opt
    #}
    #
    #pickle.dump(best_res, open(os.path.join(opt["exp"]["results_dir"], f'best_res_{opt["exp"]["exp_name"]}_{opt["conf_name"]}.pkl'), 'wb'))

    return {'opt': opt, 'best': history[best_epoch]}
