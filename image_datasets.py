from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import save_image
import PIL
import os
import numpy as np

    
def _list_image_files(data_dir):
    results = []
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            ext = file.split(".")[-1]
            if "." in file and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                # Construct the full path of the file
                full_path = os.path.join(root, file)
                results.append(full_path)
    return results

def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

class ImageLoader(Dataset):
    def __init__(self, image_paths, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform
      
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def create_loader(data_dir, img_size, batch_size):
    all_files = _list_image_files(data_dir)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = ImageLoader(all_files, transform)
    loader = iter(DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=InfiniteSamplerWrapper(dataset),
        num_workers=4, pin_memory=True
    ))
    return loader
