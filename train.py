'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Dec 2023
'''
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
import warnings
from copy import deepcopy
from collections import OrderedDict
import argparse
from fid import get_fid
from image_datasets import create_loader
from config import config
from dit import DiT
from utils import * 
from diff_utils import *

warnings.filterwarnings("ignore")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def train(model_dir, data_dir, fid_real_dir, 
          iter_interval, fid_interval, conf):
    if fid_real_dir == None:
        fid_real_dir = data_dir
    img_size = conf.img_size
    batch_size = conf.batch_size
    lr = conf.lr
    dim = conf.dim
    ema_decay = conf.ema_decay
    patch_size = conf.patch_size
    depth = conf.depth
    heads = conf.heads
    mlp_dim = conf.mlp_dim
    k = conf.k
    fid_batch_size = conf.fid_batch_size
    gen_batch_size = conf.gen_batch_size
    steps = conf.steps
    n_fid_real = conf.n_fid_real
    n_fid_gen = conf.n_fid_gen
    n_iter = conf.n_iter
    plot_shape = conf.plot_shape
    fig_size = conf.fig_size

    # dataset
    train_loader = create_loader(
        data_dir, img_size, batch_size
    )
    
    # model
    model = DiT(img_size, dim, patch_size,
            depth, heads, mlp_dim, k)
    diffusion = Diffusion()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # create ema
    ema = deepcopy(model).to(device)  
    requires_grad(ema, False)
    
    # logs and ckpt config
    gen_dir = os.path.join(model_dir, 'fid')
    log_img_dir = os.path.join(model_dir, 'log_img')
    log_dir = os.path.join(model_dir, 'log_dir')
    writer = SummaryWriter(log_dir)
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(log_img_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    last_ckpt = os.path.join(model_dir, './last_ckpt.pt')
    best_ckpt = os.path.join(model_dir, './best_ckpt.pt')
    
    if os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt)
        start_iter = ckpt['iter'] + 1 # start from iter + 1
        best_fid = ckpt['best_fid']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        print(f'Checkpoint restored at iter {start_iter}; ' 
                f'best FID: {best_fid}')
    else:
        start_iter = 1
        best_fid = 1000. # init with big value
        print(f'New model')

    # plot shape
    sz = (plot_shape[0] * plot_shape[1], 3, img_size, img_size)

    # train
    start = time.time()
    train_loss = 0.0
    update_ema(ema, model, decay=ema_decay) 
    model.train()
    ema.eval()  # EMA model should always be in eval mode
    for idx in range(n_iter):
        i = idx + start_iter
        inputs = next(train_loader)
        inputs = inputs.to(device)
        xt, t, target = diffusion.diffuse(inputs)
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = model(xt, t)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        update_ema(ema, model)
        train_loss += loss.item()

        if i % iter_interval == 0:
            # plot
            gen_batch = diffusion.sample(ema, sz, steps=steps)
            plot_path = os.path.join(log_img_dir, f'{i:04d}.png')
            plot_batch(
                deprocess(gen_batch), plot_shape, 
                fig_size, plot_path
            )
            # metrics
            train_loss /= iter_interval
            print(f'Time for iter {i} is {time.time()-start:.4f}'
                        f'sec Train loss: {train_loss:.4f}')
            writer.add_scalar('train_loss_iter', train_loss, i)
            writer.add_scalar('train_loss_n_img', train_loss, i * batch_size)
            writer.flush()
            train_loss = 0.0
            start = time.time()
            model.train()

        if i % fid_interval == 0:
            # fid
            print('Generating eval batches...')
            gen_batches(
                diffusion, ema, n_fid_real, gen_batch_size, 
                steps, gen_dir, img_size
            )
            fid = get_fid(
                fid_real_dir, gen_dir, n_fid_real, n_fid_gen,
                device, fid_batch_size
            )
            print(f'FID: {fid}')
            writer.add_scalar('FID_iter', fid, i)
            writer.add_scalar('FID_n_img', fid, i * batch_size)
            writer.flush()

            # ckpt
            ckpt_data = {
                'iter': i,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'opt': optimizer.state_dict(),
                'fid': fid,
                'best_fid': min(fid, best_fid),
                'train_loss': train_loss
            }
            
            torch.save(ckpt_data, last_ckpt)
            print(f'Checkpoint saved at iter {i}')
            
            if fid <= best_fid:
                torch.save(ckpt_data, best_ckpt)
                best_fid = fid
                print(f'Best checkpoint saved at iter {i}')
                           
            start = time.time()
            model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model_1')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fid_real_dir', type=str, default=None)
    parser.add_argument('--iter_interval', type=int, default=100)
    parser.add_argument('--fid_interval', type=int, default=100)
    args = parser.parse_args()

    conf = Config(config, args.model_dir)
    train(
        args.model_dir, args.data_dir, args.fid_real_dir, 
        args.iter_interval, args.fid_interval, conf
    )


if __name__ == '__main__':
    main()
