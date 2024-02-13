import models.graphs as graphs
from models.TINGnn import EncodeProcessDecode, GraphPooling, build_mlp
import math
from tqdm import tqdm
import hydra
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from utils import utils
from utils.utils import coarse_graining
from models.simulator import LearnedSimulator

from torch.optim.lr_scheduler import _LRScheduler



class LearnedSimulatorDiff(LearnedSimulator):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self._graph_network = EncodeProcessDecode(
                node_dim=3*(self._sequence_len-1) + 128, #+ 1, # we use sequence_len -1 + 1=use_weights for training
                # node_dim=3*(sequence_len-1) + 32 + 1,
                edge_dim=3 + 1 + self.edge_type_embedding_size,
                out_dim=6, **self.graph_network_kwargs)

    def forward(self, position_seq, next_pos, particle_types, n_node, 
                bonds=None, bond_types=None, cg_bonds=None, rgs=None,
                n_cg_node=None, cluster=None, keypoint=None):
        
        if self.particle_masses.device != position_seq.device:
            self.particle_masses = self.particle_masses.to(position_seq.device)
        
        weights = self.particle_masses[particle_types]
        particle_type_embeddings = self._encoder_preprocessor(particle_types, n_node, bonds, bond_types, weights)

        use_keypoint_cg = False
        # coarse graining.
        if self.use_coarse_graining:
            (position_seq, next_pos, particle_type_embeddings, weights) = coarse_graining(
                position_seq, next_pos, particle_type_embeddings, weights, use_keypoint_cg, cluster, keypoint)
            n_node = n_cg_node
            bonds = cg_bonds  # only operate at coarse-level after this point.

        noise_level = torch.randint(0, 10, (n_node.shape[0],), device=n_node.device)
        sampled_sigmas = self.sigmas[noise_level.repeat_interleave(n_node)].view(-1, 1, 1)
        sampled_noise = utils.get_position_noise(position_seq, 
                                                 sampled_sigmas, use_rw=False)
        noisy_position_seq = position_seq + sampled_noise
        noisy_next_pos = next_pos #+ sampled_noise[:, -1]

        # sampled_sigmas, noisy_next_pos_2 = self.sample_noisy_pos(noisy_next_pos, n_node)
        # target = (-1/(sampled_sigmas.pow(2)) * (noisy_next_pos_2 - noisy_next_pos)).squeeze()

        input_graph = self._dynamics_preprocessor(noisy_position_seq, particle_type_embeddings, n_node, 
                                                  bonds, bond_types, weights=weights)



        acceleration, history_latent_graph = self._graph_network(input_graph, return_latent=True)

#################################################################################################################
        acc_mean, acc_std = torch.split(acceleration, 3, dim=-1)
        acc_std =  0.1 + self.soft_plus(acc_std) # self.hparams.min_std + self.soft_plus(acc_std)

        acc_dist = Normal(acc_mean, acc_std)
        target_acceleration = self._inverse_decoder_postprocessor(noisy_next_pos, noisy_position_seq)
        acc_loss = -acc_dist.log_prob(target_acceleration).mean()

        # fit rgs residual and compute loss.
        # if self.hparams.property_net_hparams:
        prop_pred = self.predict_prop(history_latent_graph, noisy_position_seq[:, -1], weights, n_cg_node)
        property_loss = (prop_pred - rgs[:, -1][:, None]).pow(2).mean()
        # else:
        #     property_loss = 0
        loss = acc_loss + property_loss

        return acc_mean, target_acceleration, position_seq, loss
    

    def sample_noisy_pos(self, pos, n_node):
        noise_level = torch.randint(0, 20, (n_node.shape[0],), 
                                    device=n_node.device)
        sampled_sigmas = self.sigmas[noise_level.repeat_interleave(n_node)].view(-1, 1)
        added_noise = sampled_sigmas * torch.randn_like(pos)
        noisy_pos = pos + added_noise
        return sampled_sigmas, noisy_pos
    
    def predict(self, position_seq, particle_type_embeddings, 
                n_node, bonds=None, bond_types=None, weights=None):
        
        input_graph = self._dynamics_preprocessor(position_seq, particle_type_embeddings, n_node, 
                                                 bonds, bond_types, weights=weights)

        acceleration, history_latent_graph = self._graph_network(input_graph, return_latent=True)
        acc_mean, acc_std = torch.split(acceleration, 3, dim=-1)
        acc_std = 0.1 + self.soft_plus(acc_std)
        acc_dist = Normal(acc_mean, acc_std)
    
        acc_pred = acc_dist.sample()
        new_position = self._decoder_postprocessor(acc_pred, position_seq)

        prop_pred = self.predict_prop(history_latent_graph, position_seq[:, -1], weights, n_node)

        return new_position, prop_pred

    def _decoder_postprocessor(self, acceleration, position_seq):
        """apply <acceleration> to <position_seq> to get next position."""
        most_recent_pos = position_seq[:, -1]
        most_recent_velocity = most_recent_pos - position_seq[:, -2]
        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_pos = most_recent_pos + new_velocity  # * dt = 1
        return new_pos


    def _inverse_decoder_postprocessor(self, next_position, position_seq):
        """Inverse of `_decoder_postprocessor`."""
        previous_position = position_seq[:, -1]
        previous_velocity = previous_position - position_seq[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity
        return acceleration





class CustomScheduleLR(_LRScheduler):
  """
  the learning rate schedule used in GNS.
  """
  def __init__(self, optimizer, min_lr, decay_steps, decay_rate, last_epoch=-1, verbose=False):
    self._min_lr = min_lr
    self._decay_steps = decay_steps
    self._decay_rate = decay_rate
    super(CustomScheduleLR, self).__init__(optimizer, last_epoch, verbose)
    
  def get_lr(self):
    if not self._get_lr_called_within_step:
        warnings.warn("To get the last learning rate computed by the scheduler, "
                      "please use `get_last_lr()`.", UserWarning)
    return [(self._min_lr + (self.base_lrs[0] - self._min_lr) * 
            self._decay_rate ** (self._step_count / self._decay_steps))]

  def _get_closed_form_lr(self):
    return [(self._min_lr + (self.base_lrs[0] - self._min_lr) * 
            self._decay_rate ** (self._step_count / self._decay_steps))]
