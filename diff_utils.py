'''
Based on:
    https://github.com/NVlabs/edm/ 
    https://github.com/crowsonkb/k-diffusion
'''
import torch
from torchvision.utils import save_image
import os
from tqdm import tqdm


def get_scalings(sig, sig_data):
    s = sig ** 2 + sig_data ** 2
    # c_skip, c_out, c_in
    return sig_data ** 2 / s, sig * sig_data / s.sqrt(), 1 / s.sqrt()

def get_sigmas_karras(n, sigma_min=0.01, sigma_max=80., rho=7., device='cpu'):
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.tensor([0.])]).to(device)

class Diffusion(object):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.66):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
    def diffuse(self, y):
        device = y.device
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=device)
        sigma = (rnd_normal * self.P_mean - self.P_std).exp()
        n = torch.randn_like(y, device=device)
        c_skip, c_out, c_in = get_scalings(sigma, self.sigma_data)
        noised_input = y + n * sigma
        target = (y - c_skip * noised_input) / c_out
        return c_in * noised_input, sigma.squeeze(), target

    def sample(self, model, sz, steps=100, sigma_max=80., seed=None):
        # Set up seed and context manager
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                return self._sample_internal(model, sz, steps, sigma_max)
        else:
            return self._sample_internal(model, sz, steps, sigma_max)

    def _sample_internal(self, model, sz, steps, sigma_max):
        device = next(model.parameters()).device
        model.eval()
        x = torch.randn(sz, device=device) * sigma_max
        t_steps = get_sigmas_karras(steps, device=device, sigma_max=sigma_max)
        
        for i in range(len(t_steps) - 1):
            x = self.edm_sampler(x, t_steps, i, model)   
        return x.cpu()

    @torch.no_grad()
    def edm_sampler(self, x, t_steps, i, model, s_churn=0., s_min=0., 
                    s_max=float('inf'), s_noise=1.,):
        n = len(t_steps)
        gamma = self.get_gamma(t_steps[i], s_churn, s_min, s_max, s_noise, n)
        eps = torch.randn_like(x) * s_noise
        t_hat = t_steps[i] + gamma * t_steps[i]
        if gamma > 0: 
            x_hat = x + eps * (t_hat ** 2 - t_steps[i] ** 2) ** 0.5
        else: # gamma == 0 -> x_hat = x
            x_hat = x
        d = self.get_d(model, x_hat, t_hat)
        d_cur = (x_hat - d) / t_hat
        x_next = x_hat + (t_steps[i + 1] - t_hat) * d_cur
        if t_steps[i + 1] != 0: 
            d = self.get_d(model, x_next, t_steps[i + 1])
            d_prime = (x_next - d) / t_steps[i + 1]
            d_prime = (d_cur + d_prime) / 2
            x_next = x_hat + (t_steps[i + 1] - t_hat) * d_prime
        return x_next

    def get_d(self, model, x, sig):
        c_skip, c_out, c_in = get_scalings(sig, self.sigma_data)
        sig = sig.view(-1)
        return model(x * c_in, sig) * c_out + x * c_skip

    def get_gamma(self, t_cur, s_churn, s_min, s_max, s_noise, n):
        if s_min <= t_cur <= s_max:
            return min(s_churn / (n - 1), 2 ** 0.5 - 1)
        else:
            return 0.
        
def gen_batches(diffusion, unet, n_images, batch_size, 
                steps, dir_path, img_size):
    n_batches = n_images // batch_size
    n_used_imgs = n_batches * batch_size
    sz = (batch_size, 3, img_size, img_size)

    for i in tqdm(range(n_batches)):
        start = i * batch_size
        gen_batch = (diffusion.sample(unet, sz, steps=steps) + 1.) / 2

        img_index = start
        for img in gen_batch:
            file_name = os.path.join(dir_path, f'{str(img_index)}.png')
            save_image(img, file_name)
            img_index += 1
