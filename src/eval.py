import hydra
from omegaconf import DictConfig
import os
import models.simulator as simulator
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocessing.preprocessing_utils import dict_collate_fn, load_pt
import tqdm
import preprocessing.data as data

from utils.utils import coarse_graining
from utils.visualizations import eval_plot_3d_graph, target_plot_3d_graph, create_video
from utils import utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simulate_trajectory(dataloader, model, rollout_len, CG=False):
    new_position = None
    rollout = {}
    all_positions = []
    all_positions_real = []

    # Prevent gradient calculation for efficiency
    with torch.no_grad():
        # Extract only the first batch from dataloader
        batch = next(iter(dataloader))

        # Move all batch tensor to CUDA device
        batch = {feat_args: tensor.to('cuda') for feat_args, tensor in batch.items()}

        particle_masses = torch.FloatTensor(utils.ATOM_MASSES)
        weights = particle_masses[batch['particle_types']].to(DEVICE)
        # cluster, n_node, keypoint, n_cg_node, bonds, cg_bonds, position_seq, next_pos = (batch[key] for key in 
        #                                                                   ['cluster', 'n_particle','keypoint',
        #                                                                   'n_keypoint', 'bonds', 'cg_bonds', 
        #                                                                   'position', 'target'])
        cluster, n_node, n_cg_node, bonds, cg_bonds, position_seq, next_pos = (batch[key] for key in 
                                                                          ['cluster', 'n_particle',
                                                                          'n_keypoint', 'bonds', 'cg_bonds', 
                                                                          'position', 'target'])
        
        bond_types = batch.get('bond_types')

        particle_type_embeddings = model._encoder_preprocessor(batch['particle_types'], n_node, bonds, bond_types, weights)
        processed_embeddings = model.embedding_gn(particle_type_embeddings)

        use_keypoint_cg = False
        if CG:
            (position_seq, next_pos, processed_embeddings, weights) = coarse_graining(
                position_seq, next_pos, processed_embeddings, weights, use_keypoint_cg, cluster)#, keypoint)
            n_node = n_cg_node
            bonds = cg_bonds

        # Store each position from cg_pos_seq into all_positions
        for i in range(position_seq.shape[1]):
            all_positions.append(position_seq[:, i].detach().cpu().unsqueeze(1))
            all_positions_real.append(position_seq[:, i].detach().cpu().unsqueeze(1))

        # Predict the new_position using the model
        new_position = model.predict(position_seq, processed_embeddings, 
                                     n_node, bonds, bond_types, weights)

        # Update pos_seq with the new_position
        pos_seq = torch.cat([position_seq[:, 1:], new_position.unsqueeze(1)], dim=1)

        # Append the new_position to all_positions
        all_positions.append(new_position.detach().cpu().unsqueeze(1))
        all_positions_real.append(next_pos.detach().cpu().unsqueeze(1))

        dataloader_iterator = iter(dataloader)
        # Perform a prediction for every timestep in the rollout_len
        for i in tqdm.tqdm(range(rollout_len)):
        ################################################################################################################
            batch = next(dataloader_iterator)

            # Move all batch tensor to CUDA device
            weights = particle_masses[batch['particle_types']].to(DEVICE)
            batch = {feat_args: tensor.to('cuda') for feat_args, tensor in batch.items()}
            # cluster, n_node, keypoint, n_cg_node, bonds, cg_bonds, position_seq, next_pos = (batch[key] for key in 
            #                                                                 ['cluster', 'n_particle', 'keypoint',
            #                                                                 'n_keypoint', 'bonds', 'cg_bonds', 
            #                                                                 'position', 'target'])

            cluster, n_node, n_cg_node, bonds, cg_bonds, position_seq, next_pos = (batch[key] for key in 
                                                                            ['cluster', 'n_particle', 
                                                                            'n_keypoint', 'bonds', 'cg_bonds', 
                                                                            'position', 'target'])
            particle_type_embeddings = model._encoder_preprocessor(batch['particle_types'], n_node, bonds, bond_types, weights)
            processed_embeddings = model.embedding_gn(particle_type_embeddings)
            (position_seq, next_pos, processed_embeddings, weights) = coarse_graining(
                position_seq, next_pos, processed_embeddings, weights, use_keypoint_cg, cluster)#, keypoint)
            n_node = n_cg_node
            bonds = cg_bonds
        ###############################################################################################################
            # # Plot 3D graph of the new_position
            # Predict the new_position using the model
            new_position = model.predict(pos_seq, processed_embeddings, 
                                         n_node, bonds, bond_types, weights)

            # Update pos_seq with the new_position
            pos_seq = torch.cat([pos_seq[:, 1:], new_position.unsqueeze(1)], dim=1)

            # Append the new_position to all_positions
            all_positions.append(new_position.detach().cpu().unsqueeze(1))
            all_positions_real.append(next_pos.detach().cpu().unsqueeze(1))

            # Update the 'coordinates' key in the rollout dictionary
            rollout['coordinates'] = torch.cat(all_positions, dim=1)

        for i in tqdm.tqdm(range(len(all_positions))):
            # Plot 3D graph of the new_position
            # eval_plot_3d_graph(all_positions[i].squeeze(), batch, f"eval/{str(i)}")
            target_plot_3d_graph(all_positions[i].squeeze(), batch, all_positions_real[i].squeeze(), f"eval/{str(i)}")
        # Return the rollout dictionary after each batch
        visualization_dir = os.getenv("VISUALIZATION_DIR")
        create_video(f"{visualization_dir}/eval")
        return rollout
    
@hydra.main(config_path="conf", config_name="chignolin")        
def main(cfg: DictConfig) -> None:
    rollout_len = 2000
    test_data = data.ProteinDataset(split=cfg.data.datamodule.datasets.test.split, 
                                    directory=cfg.data.dir,
                                    grouping=100)
    test_dataloader = DataLoader(test_data, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 collate_fn=dict_collate_fn, 
                                 num_workers=4)
    model_dir = os.getenv('MODEL_DIR')
    model_path = f"{model_dir}/model_overfit_19_batch_27900"
    model = torch.load(model_path)
    rollout = simulate_trajectory(test_dataloader, model, rollout_len, CG=True)
    torch.save(rollout, 'CG_new_overfit_new_rollout.pt')
    load_pt("c:/Users/ttunj/TINGnn/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations", "CG_new_overfit_new_rollout")
    
if __name__=="__main__":
    # import ipdb;ipdb.set_trace()
    main()
