import os
import time
import hydra
import numpy as np
import torch
import ot

from pathlib import Path
from torch.utils.data import DataLoader
from preprocessing.preprocessing_utils import load_data, load_data_w_idx

# import preprocessing.data as data
# from train import seed_everything

# from preprocessing.preprocessing_utils import dict_collate_fn, load_pt

def compare_rgs(cfg):
    output = torch.load("/home/ttunjic/MY_POLYMER_MODELS/rollouts/seed_1.pt")
    pred_rgs = output['rollout_prop']
    avg_pred_rgs = torch.mean(pred_rgs, dim=1)

    data_split_path = "/home/ttunjic/Code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/data/test_class2.txt"
    with open(data_split_path, 'r') as file:
      dir_names = file.readlines()

    # Remove newline characters and construct full directory paths
    dir_names = [name.strip() for name in dir_names]

    # Data names to load from the h5 file
    data_names = ['rgs']  # Replace with your actual data names

    # List to store all tensors
    avg_values = []

    for dir_name in dir_names:
        # Construct the path to the rgs.h5 file
        # h5_path = os.path.join(cfg.data.directory, dir_name, 'rgs.h5')
        h5_path = os.path.join('/home/ttunjic/Polymer_data/polymer_test_5M', dir_name, 'rgs.h5')

        # Load data from the h5 file
        data = load_data(data_names, h5_path)

        # Convert each numpy array to a PyTorch tensor and append to the list
        for array in data:
            sliced_array = array #array[30:]
            tensor = torch.tensor(sliced_array)
            avg_tensor = torch.mean(tensor)
            avg_values.append(avg_tensor.item())

    # Convert the list of averages to a PyTorch tensor
    avg_real_rgs = torch.tensor(avg_values)


    # MAE - Mean Absolute Error
    mae = torch.mean(torch.abs(avg_pred_rgs - avg_real_rgs))

    # R squared (R²)
    ss_res = torch.sum((avg_real_rgs - avg_pred_rgs) ** 2)
    ss_tot = torch.sum((avg_real_rgs - torch.mean(avg_real_rgs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # # EMD - Earth Mover's Distance
    # # Converting tensors to numpy arrays for EMD calculation
    # real_rgs_np = avg_real_rgs.numpy().reshape(-1, 1)
    # pred_rgs_np = avg_pred_rgs.numpy().reshape(-1, 1)

    # # Normalizing data for EMD
    # real_rgs_normalized = ot.utils.norm(real_rgs_np)
    # pred_rgs_normalized = ot.utils.norm(pred_rgs_np)

    # # Computing cost matrix
    # cost_matrix = ot.dist(real_rgs_normalized, pred_rgs_normalized)

    # # Computing EMD
    # emd = ot.emd2([], [], cost_matrix)

    print(f"MAE: {mae.item()}")
    print(f"R squared: {r_squared.item()}")
    print("Tin")
    # print(f"EMD: {emd}")


@hydra.main(config_path="conf", config_name="chignolin")
def main(cfg):
    compare_rgs(cfg)

if __name__=="__main__":
    import ipdb; ipdb.set_trace()
    main()
