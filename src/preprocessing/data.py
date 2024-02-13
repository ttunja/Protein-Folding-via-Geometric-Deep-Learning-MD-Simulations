import numpy as np
import torch
from pathlib import Path
from preprocessing.preprocessing_utils import load_data, load_data_w_idx

from torch.utils.data import Dataset
import utils.utils as utils
from preprocessing.clustering import metis_cluster

class MDDataset(Dataset):
  """
  Args:
    directory:         directory to data.
    split:             subset of data to use.
    seq_len:           number of frames of each sampled data point.
    dilation:          the time interval between each two sampled frame. 
    traj_len:          length of the whole trajectory to be loaded.
  """
  def __init__(self, 
               directory, 
               split, 
               mode='train',
               seq_len=30, 
               dilation=1,
               grouping=1, 
               traj_len=10000,
               recursive_metis=False
               ):
    if isinstance(directory, str):
      directory = Path(directory)
    self.directory = directory
    
    assert mode in ['rollout', 'train', 'oneshot']
    self.mode = mode
    
    self.seq_len = seq_len
    if self.mode == 'train':
      # +1 for count the target pos
      self.seq_len += 1
    self.dilation = dilation
    self.grouping = grouping
    self.recursive_metis = recursive_metis
        
    with open(split, 'r') as f:
      traj_index = f.read().splitlines()
    self.traj_index = sorted([self.directory / traj for traj in traj_index])
    self.n_rollouts = len(self.traj_index)
    
    self.traj_len = traj_len
    self.offset = self.traj_len - self.seq_len*self.dilation + 1
  
  def get_file_indices(self, idx):
    if self.mode == 'rollout':
      idx_rollout = idx
      st_idx = 0
      ed_idx = st_idx + self.seq_len * self.dilation
    elif self.mode == 'train':
      idx_rollout = idx // self.offset
      st_idx = idx % self.offset
      ed_idx = st_idx + self.seq_len * self.dilation
    elif self.mode == 'oneshot':
      # oneshot model only uses the particle types.
      idx_rollout = idx
      st_idx = 0
      ed_idx = st_idx + self.traj_len
    return idx_rollout, st_idx, ed_idx

  def get_cg_res(self, tensor_dict):
    row, col = tensor_dict['bonds'][:, 0], tensor_dict['bonds'][:, 1]
    cluster = tensor_dict['residue_num']
    if int(torch.min(tensor_dict['residue_num']))!=0:
      cluster = cluster-int(torch.min(tensor_dict['residue_num']))

    num_cg_nodes = int(torch.max(cluster))+1
    # write keypoints as how many atoms each residue has
    boundary_bonds = (cluster[row] != cluster[col])
    cg_bonds = torch.cat([cluster[row[boundary_bonds]][:, None], 
                        cluster[col[boundary_bonds]][:, None]], dim=1)
    cg_bonds = torch.unique(cg_bonds, dim=0)
    if num_cg_nodes==35:
      new_bonds = torch.tensor([[22, 23], [23, 22], [27, 28], [28, 27]])
      cg_bonds = torch.cat((cg_bonds, new_bonds), dim=0)
  
    return {
        # 'keypoint': keypoints,
        'n_keypoint': torch.LongTensor([num_cg_nodes]),
        'cluster': cluster,
        'cg_bonds': cg_bonds,
        'n_cg_bond': torch.LongTensor([len(cg_bonds)])
          }
  
  def get_cg_stats(self, tensor_dict):
    num_cg_nodes = round(int(tensor_dict['n_particle']) / self.grouping)
    row, col = tensor_dict['bonds'][:, 0], tensor_dict['bonds'][:, 1]
    cluster, keypoints = metis_cluster(num_cg_nodes, tensor_dict['n_particle'], 
                                      row, col, recursive=self.recursive_metis)
    boundary_bonds = (cluster[row] != cluster[col])
    cg_bonds = torch.cat([cluster[row[boundary_bonds]][:, None], 
                          cluster[col[boundary_bonds]][:, None]], dim=1)
    cg_bonds = torch.unique(cg_bonds, dim=0)
    
    return {
        'keypoint': keypoints,
        'n_keypoint': torch.LongTensor([num_cg_nodes]),
        'cluster': cluster,
        'cg_bonds': cg_bonds,
        'n_cg_bond': torch.LongTensor([len(cg_bonds)])
          }

  def __len__(self):
    if self.mode in ['rollout', 'oneshot']:
      return self.n_rollouts
    else:
      return self.n_rollouts * self.offset
    
  def __getitem__(self, idx):
    raise NotImplementedError
  
class ProteinDataset(MDDataset):
  """
  Fields:
    position: T x N x 3
    rgs: T x 1
    bond_indices: (N-1) x 2
    particle_types: N x 1
  """
    
  def __getitem__(self, idx):
    idx_rollout, st_idx, ed_idx = self.get_file_indices(idx)
    select_idx = np.arange(st_idx, ed_idx, self.dilation, dtype=np.int64)
    tensor_dict = {}
    
    ptype = load_data(['particle_type'], self.traj_index[idx_rollout] / 'ptype.h5')[0]
    # TODO comment out
    # non_hydrogen_ind = list(np.where(ptype!=1)[0])
    # ptype = ptype[non_hydrogen_ind]
    rgs = load_data_w_idx(['rgs'], self.traj_index[idx_rollout] / 'rgs.h5', select_idx)[0]
    bonds = load_data(['bond_indices'], self.traj_index[idx_rollout] / 'bond.h5')[0]
    # TODO comment out
    # new_bonds = []
    # all_values = []
    # real_bonds = []
    # for k in bonds:
    #   if (k[0] and k[1]) in non_hydrogen_ind and (k[0] or k[1]) in non_hydrogen_ind:
    #     all_values.append(k[0])
    #     all_values.append(k[1])
    #     new_bonds.append(list(k))
    # all_values = list(set(all_values))
    # for bond in new_bonds:
    #   new_list = []
    #   new_list.append(all_values.index(bond[0]))
    #   new_list.append(all_values.index(bond[1]))
    #   real_bonds.append(new_list)

    # bonds = np.array(real_bonds)
    # # bonds = np.array([list(k) for k in  bonds if (k[0] and k[1]) in non_hydrogen_ind])
    reversed_bonds = np.concatenate([bonds[:, 1:], bonds[:, :1]], axis=1)
    bonds = np.concatenate([bonds, reversed_bonds], axis=0)
    tensor_dict.update({'bonds': np.array(bonds, dtype=np.int64),
                        'n_bond': np.array([bonds.shape[0]], dtype=np.int64)})
    
    if self.mode == 'oneshot':
      tensor_dict.update({'particle_types': np.array(ptype, dtype=np.int64),
                          'rgs': np.array(rgs, dtype=np.float32)[None, :],
                          'n_particle': np.array([ptype.shape[0]], dtype=np.int64)})
      tensor_dict = {k: torch.from_numpy(v) for k, v in tensor_dict.items()}   
      return tensor_dict
    
    positions = load_data_w_idx(['position'], self.traj_index[idx_rollout] / 'position.h5', 
                                      select_idx)[0]
    # TODO comment out
    # positions = positions[:][:][:,non_hydrogen_ind]
    if self.mode == 'train':
      input_pos = positions[:-1]  
      input_rgs = rgs[:-1]
    else:
      input_pos = positions
      input_rgs = rgs
    target_pos = positions[-1]
    target_rgs = rgs[-1]

    tensor_dict.update({
        'position': np.array(input_pos, dtype=np.float32).transpose(1, 0, 2),  # N x T x dim
        'rgs': np.array(input_rgs, dtype=np.float32)[None, :],
        'target': np.array(target_pos, dtype=np.float32),
        'target_rgs': np.array([target_rgs], dtype=np.float32)[None, :],
        'particle_types': np.array(ptype, dtype=np.int64),
        'n_particle': np.array([target_pos.shape[0]], dtype=np.int64)
    })
    
    # clustering use torch tensors.
    tensor_dict = {k: torch.from_numpy(v) for k, v in tensor_dict.items()}
    tensor_dict.update(self.get_cg_stats(tensor_dict))

    return tensor_dict
  