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
        # TODO hyperparam
        # self.sigmas = torch.linspace(math.log(self.hparams.sigma_begin), 
        #                             math.log(self.hparams.sigma_end), 
        #                             self.hparams.sigma_level).exp()
        self.sigmas = torch.linspace(math.log(10), 
                                    math.log(0.01), 
                                    20).exp().to('cuda')
        self.sigmas.requires_grad = False
        # self.score_gn = nets.EncodeProcessDecode(
        #   node_dim=self.latent_dim+int(self.hparams.use_weights),
        #   edge_dim=self.hparams.dimension+1+self.hparams.type_emb_size,
        #   out_dim=self.hparams.dimension, 
        #   **self.hparams.score_gn_hparams)
        # TODO najvjv treba self za varables
        self.embedding_gn = EncodeProcessDecode(
                                            node_dim=self.particle_type_embedding_size + 1,
                                            edge_dim=self.edge_type_embedding_size,  
                                            output_size=128, 
                                            **self.embedding_network_kwargs)

        # TODO check node_dim, it should be 128, not 21
        self.score_gn = EncodeProcessDecode(node_dim=128+1,
                                            edge_dim=3+1+32,
                                            hidden_size=128,
                                            num_hidden_layers=2,
                                            latent_size=128, 
                                            num_message_passing_steps=7,
                                            output_size=3)
        
    def forward(self, position_seq, next_pos, particle_types, n_node, 
                bonds=None, bond_types=None, cg_bonds=None, n_cg_node=None,
                cluster=None, keypoint=None):
        
        if self.particle_masses.device != position_seq.device:
            self.particle_masses = self.particle_masses.to(position_seq.device)
        
        weights = self.particle_masses[particle_types]
        particle_type_embeddings = self._encoder_preprocessor(particle_types, n_node, bonds, bond_types, weights)
        
        # Process particle_type_embeddings through embedding_gn
        processed_embeddings = self.embedding_gn(particle_type_embeddings)

        use_keypoint_cg = False
        # coarse graining.
        if self.use_coarse_graining:
            (position_seq, next_pos, processed_embeddings, weights) = coarse_graining(
                position_seq, next_pos, processed_embeddings, weights, use_keypoint_cg, cluster, keypoint)
        n_node = n_cg_node
        bonds = cg_bonds  # only operate at coarse-level after this point.

        sampled_noise = utils.get_position_noise(position_seq, 
                                                 self._noise_std, use_rw=True)
        noisy_position_seq = position_seq + sampled_noise
        noisy_next_pos = next_pos + sampled_noise[:, -1]

        sampled_sigmas, noisy_next_pos_2 = self.sample_noisy_pos(noisy_next_pos, n_node)
        target = (-1/(sampled_sigmas.pow(2)) * (noisy_next_pos_2 - noisy_next_pos)).squeeze()

        input_graph = self._dynamics_preprocessor(noisy_position_seq, processed_embeddings, n_node, 
                                                  bonds, bond_types, weights=weights)



        acceleration, history_latent_graph = self._graph_network(input_graph, return_latent=True)

        target_acceleration = self._inverse_decoder_postprocessor(noisy_next_pos, position_seq)

        history_latent = history_latent_graph.nodes
        score_graph = self._dynamics_preprocessor(noisy_next_pos_2[:, None], history_latent, n_node, 
                                                  bonds, bond_types, weights=weights)
        
        scores = self.score_gn(score_graph)
        scores = (scores * (1. / sampled_sigmas.view(-1, 1)))

        # TODO implement this as a score
        score_loss = (0.5 * ((scores - target) ** 2) * 
                  (sampled_sigmas ** 2)).mean()

        return acceleration, target_acceleration, position_seq, score_loss
    

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
        history_latent = history_latent_graph.nodes

        new_position = self._decoder_postprocessor(acceleration, position_seq)

        # for sigma in self.sigmas:  # [-self.hparams.dyn_level:]
        #     for ld_step in range(10):#(self.hparams.step_per_sigma):
        #         score_graph = self._dynamics_preprocessor(
        #             new_position[:, None], history_latent, n_node, bonds, bond_types, weights=weights)
        #         scores = 0.00005 * self.score_gn(score_graph) / sigma
        #         langevin_noise = torch.randn_like(new_position) * math.sqrt(0.00005 * 2)
        #         new_position = new_position + scores + langevin_noise

        return new_position

    # TODO Add embedding_gnn, you need bonds,..
    def _encoder_preprocessor(self, particle_types, n_node, bonds=None, bond_types=None, weights=None):
        # Node features    
        particle_type_embeddings = self.type_embedding(particle_types)
        if not self.embedding_network_kwargs:
            return particle_type_embeddings
        else:
            assert bonds is not None, 'an embedding network should only be used when the \
                                        ground truth graph contains bonds.'                        
            senders, receivers = bonds[:, 0], bonds[:, 1]
            edge_embeddings = particle_type_embeddings[senders] + particle_type_embeddings[receivers]
            
            # TODO check with ipdb
            if bond_types is not None:
                edge_embeddings += self.bond_type_embedding(bond_types)
            
            if weights is not None:
                particle_type_embeddings = torch.cat([particle_type_embeddings, weights.view(-1, 1)], dim=-1)
                
            return graphs.GraphsTuple(
                nodes=particle_type_embeddings,
                edges=edge_embeddings,
                edge_type=None,
                coords=None,
                globals=None,
                n_node=n_node,
                senders=senders,
                receivers=receivers)
    
    def _dynamics_preprocessor(self, position_seq, particle_type_embeddings, n_node,
                               bonds=None, bond_types=None, weights=None):
        
        velocity_sequence = self.time_diff(position_seq)  # Finite-difference.
        flat_vel_seq = torch.reshape(velocity_sequence, 
                                    [velocity_sequence.shape[0], 
                                     math.prod(velocity_sequence.shape[1:])])
        if weights is not None:
            node_features = torch.cat([flat_vel_seq, particle_type_embeddings, weights.view(-1, 1)], dim=-1)
        else:
            node_features = torch.cat([flat_vel_seq, particle_type_embeddings], dim=-1)
        

        # Edge features
        most_recent_position = position_seq[:, -1]
        # Get connectivity of the graph.
        senders, receivers, displacements, distances, edge_types = utils.compute_connectivity(
                            most_recent_position, n_node, self._connectivity_radius, bonds)
        
        edge_embeddings = self.edge_type_embedding(edge_types)

        edge_features = torch.cat([displacements, distances, edge_embeddings], dim=-1)
        
        return graphs.GraphsTuple(
            nodes=node_features,
            edges=edge_features,
            edge_type=edge_types,
            coords=None,
            globals=None,
            n_node=n_node,
            senders=senders,
            receivers=receivers)

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

        # TODO Normalize acceleration (for faster convergence)
        # acceleration_stats = self._normalization_stats["acceleration"]
        # normalized_acceleration = (
        #     acceleration - acceleration_stats.mean) / acceleration_stats.std
        # return normalized_acceleration
        return acceleration


    def time_diff(self, input_sequence):
        return input_sequence[:, 1:] - input_sequence[:, :-1]
