# minDiT hyperparameters
config = {  
    'img_size': 32,
    'batch_size': 128,
    'lr': 0.001,
    'dim': 256,
    'k': 64, # linformer dim
    'patch_size': 2,
    'depth': 2, 
    'heads': 4, 
    'mlp_dim': 128, 
    'fid_batch_size': 50, # inception 
    'gen_batch_size': 100,
    'steps': 100,  # eval diff steps
    'ema_decay': 0.999,
    'n_fid_real': 2500,
    'n_fid_gen': 2500,
    'n_iter': 1000000000,
    'plot_shape': (5, 10),
    'fig_size': (5, 2.5),
}
