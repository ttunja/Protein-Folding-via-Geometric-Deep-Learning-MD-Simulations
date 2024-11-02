import models.graphs as graphs
from models.TINGnn import EncodeProcessDecode
import math
import hydra
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from utils import utils
from utils.utils import coarse_graining
from models.simulator import LearnedSimulator

STD_EPSILON = 1e-8


class LearnedSimulatorDiff(LearnedSimulator):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph_network = EncodeProcessDecode(
                node_dim=3*(self._sequence_len-1) + self.embedding_network_kwargs.output_size, #+ 1, # we use sequence_len -1 + 1=use_weights for training
                # node_dim=3*(sequence_len-1) + 32 + 1,
                edge_dim=3 + 1 + self.edge_type_embedding_size,
                output_size=6, **self.graph_network_kwargs)
        # TODO check node_dim, it should be 128, not 21
        # self.score_gn = EncodeProcessDecode(node_dim=32,#+1,
        #                                     # node_dim=128+1,
        #                                     edge_dim=3+1+32,
        #                                     # hidden_size=128,
        #                                     hidden_size=32,
        #                                     num_hidden_layers=2,
        #                                     # latent_size=128,
        #                                     latent_size=32 ,
        #                                     # num_message_passing_steps=7,
        #                                     num_message_passing_steps=3,
        #                                     output_size=3)
        
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

        # sampled_noise = utils.get_position_noise(position_seq, 
        #                                          self._noise_std, use_rw=True)
        noise_level = torch.randint(0, 10, (n_node.shape[0],), device=n_node.device)
        sampled_sigmas = self.sigmas[noise_level.repeat_interleave(n_node)].view(-1, 1, 1).to(n_node.device)
        sampled_noise = utils.get_position_noise(position_seq, 
                                                 sampled_sigmas, use_rw=False)
        noisy_position_seq = position_seq + sampled_noise
        noisy_next_pos = next_pos #+ sampled_noise[:, -1]

        input_graph = self._dynamics_preprocessor(noisy_position_seq, particle_type_embeddings, n_node, 
                                                  bonds, bond_types, weights=weights)


        acceleration, history_latent_graph = self._graph_network(input_graph, return_latent=True)

        target_acceleration = self._inverse_decoder_postprocessor(noisy_next_pos, noisy_position_seq)
        acc_mean, acc_std = torch.split(acceleration, 3, dim=-1)
        acc_std = self._noise_std + self.soft_plus(acc_std)
        
        acc_dist = Normal(acc_mean, acc_std)

        acc_loss = -acc_dist.log_prob(target_acceleration).mean()
        # TODO Here the current property loss is with real rgs and not CG rgs, isn't that a problem???
        rgs_computed = utils.compute_weighted_rgs(position_seq[:,-1], weights, n_node)
        prop_pred = self.predict_prop(history_latent_graph, noisy_position_seq[:, -1], weights, n_cg_node)
        # property_loss = (prop_pred - rgs[:, -1][:, None]).pow(2).mean()
        property_loss = (prop_pred - rgs_computed).pow(2).mean()

        loss = acc_loss + property_loss

        return acc_mean, target_acceleration, position_seq, loss
    

    def sample_noisy_pos(self, pos, n_node):
        noise_level = torch.randint(0, 10, (n_node.shape[0],), 
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
        acc_std = self._noise_std + self.soft_plus(acc_std)
        acc_dist = Normal(acc_mean, acc_std)

        if self.deterministic:
            acc_pred = acc_mean
        else:
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
