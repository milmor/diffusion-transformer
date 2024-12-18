# Diffusion Transformer
Implementation of the Diffusion Transformer model in the paper:

> [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). 

<img src="./images/ldt.png" width="450px"></img>

See [here](https://github.com/facebookresearch/DiT) for the official Pytorch implementation.


## Dependencies
- Python 3.9
- Pytorch 2.1.1


## Training Diffusion Transformer
Use `--data_dir=<data_dir>` to specify the dataset path.
```
python train.py --data_dir=./data/
```


## Samples
Sample output from minDiT (39.89M parameters) on CIFAR-10:

<img src="./images/diff_cifar.png" width="550px"></img>

Sample output from minDiT on CelebA:

<img src="./images/diff_celeba64.png" width="650px"></img>

More samples:

<img src="./images/mindit_cifar.gif" width="650px"></img>
<img src="./images/mindit_celeba64.gif" width="550px"></img>

## Hparams setting
Adjust hyperparameters in the `config.py` file.

Implementation notes:
- minDiT is designed to offer reasonable performance using a single GPU (RTX 3080 TI).
- minDiT largely follows the original DiT model.
- DiT Block with adaLN-Zero.
- Diffusion Transformer with [Linformer](https://arxiv.org/abs/2006.04768) attention.
- [EDM](https://arxiv.org/abs/2206.00364) sampler.
- [FID](https://arxiv.org/abs/1706.08500) evaluation.


## todo
- Add Classifier-Free Diffusion Guidance and conditional pipeline.
- Add Latent Diffusion and Autoencoder training.
- Add generate.py file.


## Licence
MIT
