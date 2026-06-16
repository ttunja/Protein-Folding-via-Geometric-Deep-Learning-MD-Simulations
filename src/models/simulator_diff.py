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
        # node_dim = velocity_rnn_out (87) + embedding_gn output (128)
        # edge_dim = edges_rnn_out (4) + edge_type_embedding (32)
        # output_size = 6: acc_mean (3) + acc_std (3)
        self._graph_network = EncodeProcessDecode(
                node_dim=3*(self._sequence_len-1) + self.embedding_network_kwargs.output_size,
                edge_dim=3 + 1 + self.edge_type_embedding_size,
                output_size=6,
                equivariant=self.graph_network_kwargs.get('equivariant', False),
                **{k: v for k, v in self.graph_network_kwargs.items() if k != 'equivariant'})
        
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
        # position_seq is already CG at this point (coarse_graining applied above)
        rgs_computed = utils.compute_weighted_rgs(position_seq[:,-1], weights, n_node)
        prop_pred = self.predict_prop(history_latent_graph, noisy_position_seq[:, -1], weights, n_cg_node)
        property_loss = (prop_pred - rgs_computed).pow(2).mean()

        loss = acc_loss + property_loss

        return acc_mean, target_acceleration, position_seq, loss
    

    def forward_multistep(self, position_seq, target_seq, particle_types, n_node,
                          bonds=None, bond_types=None, cg_bonds=None, rgs=None,
                          n_cg_node=None, cluster=None, keypoint=None):
        """Multi-step rollout training. Unrolls K steps and accumulates loss.
        target_seq: N x K x 3 (K future target positions after coarse-graining)
        """
        if self.particle_masses.device != position_seq.device:
            self.particle_masses = self.particle_masses.to(position_seq.device)

        weights = self.particle_masses[particle_types]
        particle_type_embeddings = self._encoder_preprocessor(particle_types, n_node, bonds, bond_types, weights)

        use_keypoint_cg = False
        if self.use_coarse_graining:
            K = target_seq.shape[1]
            # CG all targets at once using atom-level weights before they get overwritten
            atom_weights = weights
            cg_targets = []
            for k in range(K):
                cg_t, _ = utils.compute_com(target_seq[:, k:k+1, :], atom_weights, cluster)
                cg_targets.append(cg_t.squeeze(1))
            cg_target_seq = torch.stack(cg_targets, dim=1)  # n_cg x K x 3

            # Now CG position_seq (this overwrites weights to CG-level)
            (cg_pos_seq, _, particle_type_embeddings, weights) = coarse_graining(
                position_seq, target_seq[:, 0, :], particle_type_embeddings, atom_weights,
                use_keypoint_cg, cluster, keypoint)
        else:
            cg_pos_seq = position_seq
            cg_target_seq = target_seq
            K = target_seq.shape[1]

        n_node = n_cg_node
        bonds = cg_bonds

        total_loss = 0.0
        current_pos_seq = cg_pos_seq
        K = cg_target_seq.shape[1]

        for k in range(K):
            # add noise to current position sequence
            noise_level = torch.randint(0, 10, (n_node.shape[0],), device=n_node.device)
            sampled_sigmas = self.sigmas[noise_level.repeat_interleave(n_node)].view(-1, 1, 1).to(n_node.device)
            sampled_noise = utils.get_position_noise(current_pos_seq, sampled_sigmas, use_rw=False)
            noisy_pos_seq = current_pos_seq + sampled_noise

            input_graph = self._dynamics_preprocessor(noisy_pos_seq, particle_type_embeddings,
                                                      n_node, bonds, bond_types, weights=weights)
            acceleration, latent = self._graph_network(input_graph, return_latent=True)

            # target acceleration for this step
            target_acc = self._inverse_decoder_postprocessor(cg_target_seq[:, k], noisy_pos_seq)

            acc_mean, acc_std = torch.split(acceleration, 3, dim=-1)
            acc_std = self._noise_std + self.soft_plus(acc_std)
            acc_dist = Normal(acc_mean, acc_std)
            step_loss = -acc_dist.log_prob(target_acc).mean()

            # property loss
            rgs_computed = utils.compute_weighted_rgs(current_pos_seq[:, -1], weights, n_node)
            prop_pred = self.predict_prop(latent, noisy_pos_seq[:, -1], weights, n_cg_node)
            step_loss += (prop_pred - rgs_computed).pow(2).mean()

            total_loss += step_loss

            # advance: predicted next position becomes part of the sequence
            if self.deterministic:
                next_pos = self._decoder_postprocessor(acc_mean, current_pos_seq)
            else:
                next_pos = self._decoder_postprocessor(acc_dist.sample(), current_pos_seq)
            current_pos_seq = torch.cat([current_pos_seq[:, 1:], next_pos.unsqueeze(1)], dim=1)

        return acc_mean, target_acc, cg_pos_seq, total_loss / K

    def sample_noisy_pos(self, pos, n_node):
        noise_level = torch.randint(0, 10, (n_node.shape[0],), 
                                    device=n_node.device)
        sampled_sigmas = self.sigmas[noise_level.repeat_interleave(n_node)].view(-1, 1)
        added_noise = sampled_sigmas * torch.randn_like(pos)
        noisy_pos = pos + added_noise
        return sampled_sigmas, noisy_pos
    
    def predict(self, position_seq, particle_type_embeddings,
                n_node, bonds=None, bond_types=None, weights=None, max_vel=None):

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
        new_position = self._decoder_postprocessor(acc_pred, position_seq, max_vel=max_vel)

        prop_pred = self.predict_prop(history_latent_graph, position_seq[:, -1], weights, n_node)

        return new_position, prop_pred

    def _decoder_postprocessor(self, acceleration, position_seq, max_vel=None):
        """apply <acceleration> to <position_seq> to get next position.
        If max_vel is set, clamp velocity magnitude to prevent runaway beads."""
        most_recent_pos = position_seq[:, -1]
        most_recent_velocity = most_recent_pos - position_seq[:, -2]
        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        if max_vel is not None:
            vel_mag = new_velocity.norm(dim=-1, keepdim=True)
            scale = torch.clamp(max_vel / (vel_mag + 1e-8), max=1.0)
            new_velocity = new_velocity * scale
        new_pos = most_recent_pos + new_velocity  # * dt = 1
        return new_pos


    def _inverse_decoder_postprocessor(self, next_position, position_seq):
        """Inverse of `_decoder_postprocessor`."""

        previous_position = position_seq[:, -1]
        previous_velocity = previous_position - position_seq[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity
        return acceleration
