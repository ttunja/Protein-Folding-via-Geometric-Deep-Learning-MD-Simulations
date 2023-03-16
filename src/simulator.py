import graph_nets as gn
from TINGnn import EncodeProcessDecode
import math
import hydra
import torch
import torch.nn as nn
import utils

STD_EPSILON = 1e-8


class LearnedSimulator(nn.Module):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(
        self,
        connectivity_radius,
        graph_network_kwargs,
        # normalization_stats,
        num_particle_types,
        num_bond_types,
        particle_type_embedding_size,
        edge_type_embedding_size,
        particle_types,
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

        self._connectivity_radius = connectivity_radius
        self._num_bond_types = num_bond_types
        self._particle_types = particle_types
        self.particle_masses = torch.FloatTensor(utils.ATOM_MASSES)
        # self._normalization_stats = normalization_stats
        self._sequence_len = sequence_len
        self._noise_std = noise_std
        
        self._graph_network = EncodeProcessDecode(
                node_dim=3*(sequence_len-1) + particle_type_embedding_size,
                edge_dim=3 + 1 + edge_type_embedding_size,
                output_size=3, **graph_network_kwargs)

        self.type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        
        # types of fine level bonds. set <num_btypes==0> to disable bond type information.
        if num_bond_types > 0:
            self.bond_type_embedding = nn.Embedding(num_bond_types, particle_type_embedding_size)
            nn.init.xavier_uniform_(self.bond_type_embedding.weight)
        
        self.edge_type_embedding = nn.Embedding(2, edge_type_embedding_size)  # 2 types: real bonds vs radius edge.   
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)
        
    def forward(self, position_seq, next_pos, particle_types, n_node, 
                bonds=None, bond_types=None):
        
        if self.particle_masses.device != position_seq.device:
            self.particle_masses = self.particle_masses.to(position_seq.device)
        
        sampled_noise = utils.get_position_noise(position_seq, 
                                                 self._noise_std, use_rw=True)
        noisy_position_seq = position_seq + sampled_noise
        noisy_next_pos = next_pos + sampled_noise[:, -1]
        weights = self.particle_masses[self._particle_types]
        input_graph = self._encoder_preprocessor(noisy_position_seq, particle_types, n_node, 
                                                 bonds, bond_types, weights=weights)

        acceleration = self._graph_network(input_graph)

        target_acceleration = self._inverse_decoder_postprocessor(noisy_next_pos, position_seq)

        return acceleration, target_acceleration
    
    def predict(self, position_seq, particle_types, 
                n_node, bonds=None, bond_types=None):
        
        weights = self.particle_masses[self._particle_types]
        input_graph = self._encoder_preprocessor(position_seq, particle_types, n_node, 
                                                 bonds, bond_types, weights=weights)

        acceleration = self._graph_network(input_graph)

        new_position = self._decoder_postprocessor(acceleration, position_seq)

        return new_position

    def _encoder_preprocessor(self, position_seq, particle_types, n_node, 
                              bonds=None, bond_types=None, weights=None):

        # Node features    
        particle_type_embeddings = self.type_embedding(particle_types)

        velocity_sequence = self.time_diff(position_seq)  # Finite-difference.
        flat_vel_seq = torch.reshape(velocity_sequence, 
                                    [velocity_sequence.shape[0], 
                                     math.prod(velocity_sequence.shape[1:])])
        if weights is not None:
            node_features = torch.cat([flat_vel_seq, particle_type_embeddings, weights], dim=-1)
        else:
            node_features = torch.cat([flat_vel_seq, particle_type_embeddings], dim=-1)
        
        # Edge features
        # NOT SURE            
        # senders, receivers = bonds[:, 0], bonds[:, 1]
        # edge_embeddings = ptype_embeddings[senders] + ptype_embeddings[receivers]
        
        if bond_types is not None and self._num_bond_types > 0:
            edge_embeddings = self.bond_type_embedding(bond_types)
        most_recent_position = position_seq[:, -1]
        # Get connectivity of the graph.
        senders, receivers, displacements, distances, edge_types = utils.compute_connectivity(
                            most_recent_position, n_node, self._connectivity_radius, bonds)
        edge_embeddings += self.edge_type_embedding(edge_types)
        edge_features = torch.cat([displacements, distances, edge_embeddings], dim=-1)
        
        return gn.graphs.GraphsTuple(
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


    def time_diff(input_sequence):
        return input_sequence[:, 1:] - input_sequence[:, :-1]



###########################################################################################
    # def get_predicted_and_target_normalized_accelerations(
    #     self, next_position, position_seq_noise, position_seq,
    #     n_particles_per_example, global_context=None, particle_types=None):  # pylint: disable=g-doc-args
    #     """Produces normalized and predicted acceleration targets.
    #     Args:
    #     next_position: Tensor of shape [num_particles_in_batch, num_dimensions]
    #         with the positions the model should output given the inputs.
    #     position_seq_noise: Tensor of the same shape as `position_seq`
    #         with the noise to apply to each particle.
    #     position_seq, n_node, global_context, particle_types: Inputs to the
    #         model as defined by `_build`.
    #     Returns:
    #     Tensors of shape [num_particles_in_batch, num_dimensions] with the
    #         predicted and target normalized accelerations.
    #     """

    #     # Add noise to the input position sequence.
    #     noisy_position_seq = position_seq + position_seq_noise

    #     # Perform the forward pass with the noisy position sequence.
    #     input_graphs_tuple = self._encoder_preprocessor(
    #         noisy_position_seq, n_particles_per_example, global_context,
    #         particle_types)
    #     predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

    #     # Calculate the target acceleration, using an `adjusted_next_position `that
    #     # is shifted by the noise in the last input position.
    #     next_position_adjusted = next_position + position_seq_noise[:, -1]
    #     target_normalized_acceleration = self._inverse_decoder_postprocessor(
    #         next_position_adjusted, noisy_position_seq)
    #     # As a result the inverted Euler update in the `_inverse_decoder` produces:
    #     # * A target acceleration that does not explicitly correct for the noise in
    #     #   the input positions, as the `next_position_adjusted` is different
    #     #   from the true `next_position`.
    #     # * A target acceleration that exactly corrects noise in the input velocity
    #     #   since the target next velocity calculated by the inverse Euler update
    #     #   as `next_position_adjusted - noisy_position_seq[:,-1]`
    #     #   matches the ground truth next velocity (noise cancels out).

    #     return predicted_normalized_acceleration, target_normalized_acceleration

# TODO FINISH _encoder_preprocessor(), _init_, add noise (utils), provjeri embeddings jos jednom
# TODO train.py - use hydra, 
