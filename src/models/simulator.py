import models.graphs as graphs
from models.TINGnn import EncodeProcessDecode
import math
import hydra
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from utils import utils
from utils.utils import coarse_graining

STD_EPSILON = 1e-8


class LearnedSimulator(nn.Module):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(
        self,
        connectivity_radius,
        graph_network_kwargs,
        embedding_network_kwargs,
        # normalization_stats,
        num_particle_types,
        num_bond_types,
        particle_type_embedding_size,
        edge_type_embedding_size,
        sequence_len,
        noise_std
        ):
        """Inits the model.
        Args:
        num_dimensions: Dimensionality of the problem.
        connectivity_radius: Scalar with the radius of connectivity.
        graph_network_kwargs: Keyword arguments to pass to the learned part
            of the graph network `model.EncodeProcessDecode`.
        boundaries: List of 2-tuples, containing the lower and upper boundaries of
            the cuboid containing the particles along each dimensions, matching
            the dimensionality of the problem.
        normalization_stats: Dictionary with statistics with keys "acceleration"
            and "velocity", containing a named tuple for each with mean and std
            fields, matching the dimensionality of the problem.
        num_particle_types: Number of different particle types.
        particle_type_embedding_size: Embedding size for the particle type.
        name: Name of the Sonnet module.
        """
        super().__init__()

        self.soft_plus = torch.nn.Softplus()
        self._connectivity_radius = connectivity_radius
        self._num_bond_types = num_bond_types
        self.particle_masses = torch.FloatTensor(utils.ATOM_MASSES)
        # self._normalization_stats = normalization_stats
        self._sequence_len = sequence_len
        self._noise_std = noise_std
s        self.use_coarse_graining = False #(self.hparams.cg_level > 1) or ('use_keypoint_for_cg' in self.hparams and self.hparams.use_keypoint_for_cg)
        
        self.embedding_network_kwargs = embedding_network_kwargs
        self.particle_type_embedding_size = particle_type_embedding_size
        self.edge_type_embedding_size = edge_type_embedding_size

        self._graph_network = EncodeProcessDecode(
                node_dim=3*(sequence_len-1) + particle_type_embedding_size + 1, # we use sequence_len -1 + 1=use_weights for training
                edge_dim=3 + 1 + edge_type_embedding_size,
                output_size=1, **graph_network_kwargs)

        self.type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        
        # types of fine level bonds. set <num_btypes==0> to disable bond type information.
        if num_bond_types > 0:
            self.bond_type_embedding = nn.Embedding(num_bond_types, particle_type_embedding_size)
            nn.init.xavier_uniform_(self.bond_type_embedding.weight)
        
        self.edge_type_embedding = nn.Embedding(2, edge_type_embedding_size)  # 2 types: real bonds vs radius edge.   
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)
        
    def forward(self, position_seq, next_pos, particle_types, n_node, 
                bonds=None, bond_types=None, cg_bonds=None, n_cg_node=None,
                cluster=None, keypoint=None, phi_atoms=None, psi_atoms=None, 
                phi_angles=None, psi_angles=None, target_phi_angles=None, target_psi_angles=None):
        
        if self.particle_masses.device != position_seq.device:
            self.particle_masses = self.particle_masses.to(position_seq.device)
        
        weights = self.particle_masses[particle_types]
        particle_type_embeddings = self._encoder_preprocessor(particle_types)
        
        use_keypoint_cg = False
        # coarse graining.
        if self.use_coarse_graining:
            (position_seq, next_pos, particle_type_embeddings, weights) = coarse_graining(
                position_seq, next_pos, particle_type_embeddings, weights, use_keypoint_cg, cluster, keypoint)
        n_node = n_cg_node
        bonds = cg_bonds  # only operate at coarse-level after this point.

        # sampled_noise = utils.get_position_noise(position_seq, 
        #                                          self._noise_std, use_rw=True)

        noisy_position_seq = position_seq + sampled_noise
        noisy_next_pos = next_pos + sampled_noise[:, -1]

        input_graph = self._dynamics_preprocessor(noisy_position_seq, particle_type_embeddings, n_node, 
                                                 bonds, bond_types, weights=weights)



        processed_graph = self._graph_network(input_graph)

        # TODO write this function that will actually be a MLP that takes phi atoms and psi atoms and predict respective angles
        torsion_angles = self.predict_torsion_angles(processed_graph, phi_atoms, psi_atoms, phi_angles, psi_angles)

        target_acceleration = self._inverse_decoder_postprocessor(noisy_next_pos, position_seq)

        return acceleration, target_acceleration, position_seq
    
    def predict(self, position_seq, particle_type_embeddings, 
                n_node, bonds=None, bond_types=None, weights=None):
        
        input_graph = self._dynamics_preprocessor(position_seq, particle_type_embeddings, n_node, 
                                                 bonds, bond_types, weights=weights)

        acceleration = self._graph_network(input_graph)

        new_position = self._decoder_postprocessor(acceleration, position_seq)

        return new_position

    def _encoder_preprocessor(self, particle_types):
        # Node features    
        particle_type_embeddings = self.type_embedding(particle_types)
        return particle_type_embeddings
    
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
        # NOT SURE            
        # senders, receivers = bonds[:, 0], bonds[:, 1]
        # edge_embeddings = ptype_embeddings[senders] + ptype_embeddings[receivers]
        edge_embeddings = None
        if bond_types is not None and self._num_bond_types > 0:
            edge_embeddings = self.bond_type_embedding(bond_types)
        most_recent_position = position_seq[:, -1]
        # Get connectivity of the graph.
        senders, receivers, displacements, distances, edge_types = utils.compute_connectivity(
                            most_recent_position, n_node, self._connectivity_radius, bonds)
        # TODO change this part of code
        if edge_embeddings is not None:
            edge_embeddings += self.edge_type_embedding(edge_types)
        else:
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
