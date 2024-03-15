import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json


def deprocess(img):
    return img * 127.5 + 127.5

def plot_batch(batch, plot_shape=(5, 10), fig_size=(5, 1.5), save_dir=None):
    images = batch[:plot_shape[0] * plot_shape[1]]
    fig, axes = plt.subplots(
        plot_shape[0], plot_shape[1], 
        figsize=fig_size
    )
    axes = axes.flatten()
    images = images.to(torch.int32).clamp(min=0, max=255)
    for i in range(len(images)):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].axis('off')
    
    plt.subplots_adjust(
        wspace=0, hspace=0, left=0, right=1, bottom=0, top=1
    )
    if save_dir:
        plt.savefig(save_dir)
        plt.close()
    else:
        plt.show()

class Config(object):
    def __init__(self, input_dict, save_dir):
        for key, value in input_dict.items():
            setattr(self, key, value)
        file_path = os.path.join(save_dir, "config.json")

        # Check if the configuration file exists
        if os.path.exists(file_path):
            self.load_config(file_path)
        else:
            self.save_config(file_path, save_dir)

    def save_config(self, file_path, save_dir):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert input_dict to JSON and save to file
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)
        print(f'New config {file_path} saved')

    def load_config(self, file_path):
        # Load configuration from the existing file
        with open(file_path, "r") as f:
            config_data = json.load(f)

        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            setattr(self, key, value)
        print(f'Config {file_path} loaded')
