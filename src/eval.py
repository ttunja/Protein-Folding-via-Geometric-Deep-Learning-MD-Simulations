import simulator
import os
import simulator
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocessing_utils import dict_collate_fn, load_pt
import tqdm
import data

def simulate_traj(dataloader, model, rollout_len):
    new_position = None
    rollout={}
    all_positions = []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm.tqdm(dataloader)):
            for feat_args in batch.keys():
                batch[feat_args] = batch[feat_args].to('cuda')

            if new_position is None:
                for i in range(batch['position'].shape[1]):
                    all_positions.append(batch['position'][:, i].detach().cpu().unsqueeze(1))
                
                new_position = model.predict(batch['position'], batch['particle_types'], 
                                             batch['n_particle'], batch['bonds'], 
                                             batch.get('bond_types'))
                pos_seq = torch.cat([batch['position'][:, 1:], new_position.unsqueeze(1)], dim=1)
                # all_positions.append(batch['position'].detach().cpu().unsqueeze(1))
                all_positions.append(new_position.detach().cpu().unsqueeze(1))
            else:
                if batch_num:
                    error = (sum((new_position-batch['target'])**2)/166)**0.5
                    print(f"first rmse equals {error}")
                for t in tqdm.tqdm(range(rollout_len)):
                    new_position = model.predict(pos_seq, batch['particle_types'], 
                                        batch['n_particle'], batch['bonds'], 
                                        batch.get('bond_types'))
                    pos_seq = torch.cat([pos_seq[:, 1:], new_position.unsqueeze(1)], dim=1)
                    all_positions.append(new_position.detach().cpu().unsqueeze(1))
                    rollout['coordinates'] = torch.cat(all_positions, dim=1)
                
                return rollout
        
def main():
    rollout_len = 2000
    test_data = data.ProteinDataset(split="src/test.txt", 
                                        directory= "D:/MD_Data/polymer_train")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=dict_collate_fn, num_workers=2)
    model = torch.load("models/model_overfit_49_batch_900")
    rollout = simulate_traj(test_dataloader, model, rollout_len)
    torch.save(rollout, 'new_overfit_new_rollout.pt')
    load_pt("c:/Users/ttunj/TINGnn/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations", "new_overfit_new_rollout")
    
    #"C:\Users\ttunj\TINGnn\Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations\new_predicted_traj\coordinates.h5"
    # U py39 conda environmentu
# mdconvert "C:\Users\ttunj\TINGnn\Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations\scaled_new_predicted_traj\coordinates.h5" -o scaled_new_traj_longer.h5 -t "D:\PROTEIN-FOLDING-DATA\Chignolin\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein\Topology.pdb" 
    

if __name__=="__main__":
    # import ipdb;ipdb.set_trace()
    main()