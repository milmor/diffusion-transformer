import torch
from torch import nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import linalg
import PIL
import os
import time
from image_datasets import _list_image_files


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)
        model.eval()
        # antialias == True required to work
        self.preprocess = weights.transforms(antialias=True)
        self.body = create_feature_extractor(
            model, return_nodes={'avgpool': '0'})
        
        for param in self.body.parameters():
            param.requires_grad = False
            
    @torch.no_grad()    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.body(x)['0']
        x = torch.squeeze(x, [2, 3])
        return x

class FIDLoader(Dataset):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
      
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pil_image = PIL.Image.open(self.image_paths[idx])
        arr = np.array(pil_image.convert("RGB"))
        return np.transpose(arr, [2, 0, 1])

def create_fid_loader(data_dir, batch_size, n_images=2500):
    all_files = _list_image_files(data_dir)
    random_files = np.random.choice(
        all_files, size=n_images, replace=False
    )

    dataset = FIDLoader(random_files)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, shuffle=False,
                        num_workers=4)
    return loader

def get_activations(loader, inception, device, batch_size=50):
    pred_arr = np.empty((len(loader.dataset), 2048), 'float32')
    for i, inputs in enumerate(loader):
        inputs = inputs.to(device)
        start = i * batch_size
        end = start + batch_size
        pred = inception(inputs)
        pred_arr[start:end] = pred.cpu() 

    return pred_arr

def calculate_activation_statistics(images, model, device, batch_size=50):
    act = get_activations(images, model, device, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_fid(real_dir, gen_dir, n_real, n_gen, device, batch_size=50):
    inception = Inception()
    inception.to(device)
    start = time.time()
    m_file = f'm_{n_real}.npy'
    s_file = f's_{n_real}.npy'
    # Check if files exist in the directory
    m_path = os.path.join(real_dir, m_file)
    s_path = os.path.join(real_dir, s_file)
    if os.path.exists(m_path) and os.path.exists(s_path):
        # Files exist, so read them
        m1 = np.load(m_path)
        s1 = np.load(s_path)
        print(f"Existing files loaded: {m_path}, {s_path}")
    else:
        # Files don't exist, so create them
        real_loader = create_fid_loader(real_dir, batch_size, n_real)
        print(f'{len(real_loader.dataset)} real images')
        m1, s1 = calculate_activation_statistics(real_loader, inception, 
                                                 device, batch_size)
        np.save(m_path, m1)
        np.save(s_path, s1)
        print(f"New files created and saved: {m_path}, {s_path}")
    
    gen_loader = create_fid_loader(gen_dir, batch_size, n_gen)
    print(f'{len(gen_loader.dataset)} gen images')
    m2, s2 = calculate_activation_statistics(gen_loader, inception,
                                             device, batch_size)
        
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print(f'Time for FID is {time.time()-start:.4f} sec')
    del inception, m1, s1, m2, s2
    return fid_value
