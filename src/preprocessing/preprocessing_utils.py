import os
from pathlib import Path
import time
import copy
import numpy as np
# import torch
import h5py

def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()

def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data

def load_data_w_idx(data_names, path, select_idx):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i])[select_idx])
        data.append(d)
    hf.close()
    return data

def dict_collate_fn(batch):
    import torch
    batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
    
    # bonds add bonds offsets
    if 'bonds' in batch.keys():
        index_offsets = torch.cumsum(batch['n_bond'], -1)
        node_offsets = torch.cumsum(batch['n_particle'], -1)
        for idx in range(len(index_offsets)-1):
            batch['bonds'][index_offsets[idx]:index_offsets[idx+1]] += node_offsets[idx]
        
    # GROUPING: add cluster offsets
    if 'n_keypoint' in batch:
        if 'n_component' in batch:
            component_offsets = torch.cumsum(batch['n_component'], -1)
        index_offsets = torch.cumsum(batch['n_keypoint'], -1)
        node_offsets = torch.cumsum(batch['n_particle'], -1)
        bond_offset = torch.cumsum(batch['n_cg_bond'], -1)
        for idx in range(len(index_offsets)-1):
            if 'n_component' in batch:
                batch['component'][node_offsets[idx]:node_offsets[idx+1]] += component_offsets[idx]
            if 'keypoint' in batch:
                batch['keypoint'][index_offsets[idx]:index_offsets[idx+1]] += node_offsets[idx]
            batch['cluster'][node_offsets[idx]:node_offsets[idx+1]] += index_offsets[idx]
            batch['cg_bonds'][bond_offset[idx]:bond_offset[idx+1]] += index_offsets[idx] 
    return batch

def fix_topology(topology):

    new_top = topology.copy()

    residues = {}
    for chain in new_top.chains:
        #print chain
        for residue in chain.residues:
            resname = str(residue)
            if resname in list(residues.keys()):
                residues[resname].append(residue)
            else:
                residues[resname] = [residue]

    for resname in list(residues.keys()):
        fragments = residues[resname]
        if len(fragments) > 1:
            main_fragment = fragments[0]
            new_atom_list = []
            new_atom_list += main_fragment._atoms
            for i in range(1,len(fragments)):
                fragment = fragments[i]
                for atom in fragment.atoms:
                    atom.residue = main_fragment
                new_atom_list += fragment._atoms
                fragment._atoms = []
                fragment.chain = main_fragment.chain
            main_fragment._atoms = new_atom_list

    return new_top

def fix_traj(traj):
    time0 = time.time()
    new_traj = copy.deepcopy(traj)
    topology = new_traj.topology 

    new_top = fix_topology(topology)
    topology = new_top
    new_traj.topology = new_top 

    new_atom_sequence = [a for a in topology.atoms]
    new_index_sequence = [a.index for a in topology.atoms]

    for i in range(0, np.shape(traj.xyz)[0]):
        new_traj.xyz[i] = new_traj.xyz[i][new_index_sequence]

    for i in range(0, len(new_index_sequence)):
        new_atom_sequence[i].index = i

    time1 = time.time()
    print(time1 - time0)
    return new_traj


def load_pt(datadir:str, rollout:str):
    import torch
    df = torch.load(os.path.join(datadir, rollout+".pt"))
    data = df['coordinates'].permute(1, 0, 2).detach().cpu().numpy()
    os.makedirs(os.path.join(datadir, rollout), exist_ok=True)
    if not Path(str(os.path.join(datadir, rollout, 'bond.h5'))).exists():
        store_data(['coordinates'], [data], os.path.join(datadir, rollout, 'coordinates.h5'))
