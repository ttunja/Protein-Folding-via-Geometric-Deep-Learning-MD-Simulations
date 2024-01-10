import models.graphs as graphs
from models.TINGnn import EncodeProcessDecode, AnglePredictionNetwork
import math
import hydra
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from utils import utils
from utils.utils import coarse_graining
from preprocessing.preprocess import calculate_all_torsion_angles_for_trajectories

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

        # self.soft_plus = torch.nn.Softplus()
        self._connectivity_radius = connectivity_radius
        self._num_bond_types = num_bond_types
        self.particle_masses = torch.FloatTensor(utils.ATOM_MASSES)
        # self._normalization_stats = normalization_stats
        self._sequence_len = sequence_len
        self._noise_std = noise_std
        self.use_coarse_graining = False #(self.hparams.cg_level > 1) or ('use_keypoint_for_cg' in self.hparams and self.hparams.use_keypoint_for_cg)
        
        self.embedding_network_kwargs = embedding_network_kwargs
        self.particle_type_embedding_size = particle_type_embedding_size
        self.edge_type_embedding_size = edge_type_embedding_size

        # Example of usage in your main model
        self.angle_prediction_network_phi = AnglePredictionNetwork(input_size=4, hidden_size=32, output_size=1)
        self.angle_prediction_network_psi = AnglePredictionNetwork(input_size=4, hidden_size=32, output_size=1)

        self.embedding_gn = EncodeProcessDecode(
                node_dim=self.particle_type_embedding_size + 1,
                edge_dim=self.edge_type_embedding_size,  
                output_size=128, 
                **self.embedding_network_kwargs)

        self._graph_network = EncodeProcessDecode(
                node_dim=2*(sequence_len-1) + 128 + 1, # we use sequence_len -1 + 1=use_weights for training, 128 output size of embedding_gnn, 2*(sequence_len-1) because psi+phi angles sequences
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
                bonds=None, bond_types=None, phi_atoms=None, psi_atoms=None, 
                phi_angles=None, psi_angles=None, target_phi_angles=None, target_psi_angles=None):
        
        if self.particle_masses.device != position_seq.device:
            self.particle_masses = self.particle_masses.to(position_seq.device)
        
        weights = self.particle_masses[particle_types]
        particle_type_embeddings = self._encoder_preprocessor(particle_types, n_node, bonds, bond_types, weights)

        processed_embeddings = self.embedding_gn(particle_type_embeddings)

        sampled_noise_phi = utils.get_position_noise(phi_angles, 
                                                 self._noise_std, use_rw=True)
        sampled_noise_psi = utils.get_position_noise(psi_angles, 
                                                 self._noise_std, use_rw=True)

        noisy_phi_angles = phi_angles + sampled_noise_phi
        noisy_psi_angles = psi_angles + sampled_noise_psi
        noisy_next_phi_angle = target_phi_angles + sampled_noise_phi[:, -1]
        noisy_next_psi_angle = target_psi_angles + sampled_noise_psi[:, -1]

        input_graph = self._dynamics_preprocessor(position_seq, noisy_phi_angles, noisy_psi_angles, phi_atoms, psi_atoms, 
            	                                 processed_embeddings, n_node, 
                                                 bonds, bond_types, weights=weights)

        processed_graph = self._graph_network(input_graph)

        # TODO write this function that will actually be a MLP that takes phi atoms and psi atoms and predict respective angles
        phi_angle_pred, psi_angle_pred = self.predict_torsion_angles(processed_graph, phi_atoms, psi_atoms, n_node)

        target_acceleration = self._inverse_decoder_postprocessor(next_pos, position_seq)

        # phi_angle_target = self.compute_angular_difference(noisy_next_phi_angle, phi_angles)
        # psi_angle_target = self.compute_angular_difference(noisy_next_psi_angle, psi_angles)
        phi_angle_target = noisy_next_phi_angle
        psi_angle_target = noisy_next_psi_angle

        # new_position = self.update_positions(position_seq[:,-1], phi_atoms, psi_atoms, target_phi_angles, target_psi_angles)
        new_position = 0
        # return target_acceleration, position_seq, phi_angle_pred, psi_angle_pred, phi_angle_target, psi_angle_target, new_position
        # new_position_unsqueeze = new_position.unsqueeze(0)
        # angles = calculate_all_torsion_angles_for_trajectories(new_position_unsqueeze.cpu(), phi_atoms, psi_atoms)


        return target_acceleration, position_seq, phi_angle_pred, psi_angle_pred, phi_angle_target, psi_angle_target, new_position

    def predict(self, position_seq, particle_type_embeddings, 
                n_node, bonds=None, bond_types=None, weights=None):
        
        input_graph = self._dynamics_preprocessor(position_seq, particle_type_embeddings, n_node, 
                                                 bonds, bond_types, weights=weights)

        acceleration = self._graph_network(input_graph)

        new_position = self._decoder_postprocessor(acceleration, position_seq)

        return new_position

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
    
    def _dynamics_preprocessor(self, position_seq, phi_angles, psi_angles, phi_atoms, psi_atoms, particle_type_embeddings, n_node,
                               bonds=None, bond_types=None, weights=None):
        
        
        batch_size = n_node.shape[0]
        nodes_per_sample = n_node[0].item()  # assuming the same number of nodes per sample in the batch

        # Calculate finite differences
        phi_angles_sequence = self.time_diff(phi_angles)  # shape: [90, 30]
        psi_angles_sequence = self.time_diff(psi_angles)  # shape: [90, 30]

        # Initialize node_features for the entire batch
        node_features = torch.zeros((batch_size, nodes_per_sample, phi_angles_sequence.shape[1] + psi_angles_sequence.shape[1]), device=phi_angles_sequence.device)

        # Assign phi and psi angle features to the respective atoms
        node_idx = 0
        for b in range(batch_size):
            for idx, atom_indices in enumerate(phi_atoms):
                for atom_index in atom_indices:
                    node_features[b, atom_index, :phi_angles_sequence.shape[1]] = phi_angles_sequence[idx]

            for idx, atom_indices in enumerate(psi_atoms):
                for atom_index in atom_indices:
                    node_features[b, atom_index, phi_angles_sequence.shape[1]:] = psi_angles_sequence[idx]

        # Reshape node_features for concatenation
        node_features = node_features.view(-1, node_features.shape[-1])

        # Concatenate with particle type embeddings
        if weights is not None:
            reshaped_weights = weights.view(batch_size, nodes_per_sample, 1).view(-1, 1)
            node_features = torch.cat([node_features, particle_type_embeddings, reshaped_weights], dim=-1)
        else:
            node_features = torch.cat([node_features, particle_type_embeddings], dim=-1)


        
        
        # batch_size = n_node.shape[0]
        # # TODO 
        # # 1) it get only as large as 360 and as small as -360 
        # # 2) There should be equivalnce (invariance) between -1 and 359 degrees
        # phi_angles_sequence = self.time_diff(phi_angles)  # Finite-difference.
        # psi_angles_sequence = self.time_diff(psi_angles)  # Finite-difference.

        # node_features = torch.zeros((n_node, phi_angles_sequence.shape[1] + psi_angles_sequence.shape[1]), device=phi_angles_sequence.device)

        #  # Assign phi and psi angle features to the respective atoms
        # for idx, atom_indices in enumerate(phi_atoms):
        #     for atom_index in atom_indices:
        #         node_features[atom_index, :phi_angles_sequence.shape[1]] = phi_angles_sequence[idx]

        # for idx, atom_indices in enumerate(psi_atoms):
        #     for atom_index in atom_indices:
        #         node_features[atom_index, phi_angles_sequence.shape[1]:] = psi_angles_sequence[idx]

        # # Concatenate with particle type embeddings
        # if weights is not None:
        #     node_features = torch.cat([node_features, particle_type_embeddings, weights.view(-1, 1)], dim=-1)
        # else:
        #     node_features = torch.cat([node_features, particle_type_embeddings], dim=-1)
        

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
        # Convert angles from degrees to radians
        input_sequence_rad = torch.deg2rad(input_sequence)

        # Compute the difference in radians
        diff_rad = input_sequence_rad[:, 1:] - input_sequence_rad[:, :-1]

        # Adjust differences to account for angle wrap-around
        diff_rad = torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))

        # (Optional) Convert the differences back to degrees
        diff_deg = torch.rad2deg(diff_rad)

        return diff_deg

    def compute_angular_difference(self, next_angles, current_angles):
        # Calculate the angular difference
        angular_difference = next_angles - current_angles[:,-1]

        # Adjust the difference for angle wrap-around (360 degrees or 2π radians)
        # Ensuring the difference lies between -180 and 180 degrees
        angular_difference = (angular_difference + 180) % 360 - 180

        return angular_difference

    def predict_torsion_angles(self, processed_graph, phi_atoms, psi_atoms, n_node):
        # # Assuming processed_graph is of shape [num_nodes, feature_size]
        # # and phi_atoms/psi_atoms are lists of indices (each element is a tuple of 4 indices)

        # # Lists to hold predicted phi and psi angles
        # phi_angles = []
        # psi_angles = []

        # for angle_type, atom_indices in zip(['phi', 'psi'], [phi_atoms, psi_atoms]):
        #     for indices in atom_indices:
        #         # Extract features for the group of four atoms
        #         group_features = processed_graph[indices, :].view(-1)  # Reshape to a single feature vector

        #         # Predict the angle using a dedicated network (e.g., an MLP)
        #         angle = self.angle_prediction_network(group_features)
                
        #         # Append the predicted angle to the respective list
        #         if angle_type == 'phi':
        #             phi_angles.append(angle)
        #         elif angle_type == 'psi':
        #             psi_angles.append(angle)

        # # Convert lists of angles to tensors
        # phi_tensor = torch.stack(phi_angles)
        # psi_tensor = torch.stack(psi_angles)

        # return phi_tensor, psi_tensor

        batch_size = n_node.shape[0]
        nodes_per_sample = n_node[0].item()  # Assuming the same number of nodes per sample in the batch

        # Total number of phi and psi angles per batch
        angles_per_batch_phi = len(phi_atoms) // batch_size
        angles_per_batch_psi = len(psi_atoms) // batch_size

        # Preparing tensors to store the predicted angles for each batch
        phi_tensor = torch.zeros((batch_size, angles_per_batch_phi, 1), device=processed_graph.device)
        psi_tensor = torch.zeros((batch_size, angles_per_batch_psi, 1), device=processed_graph.device)

        # Loop over the batch
        for b in range(batch_size):
            # Calculate the starting and ending index for nodes in this sample
            start_idx = b * nodes_per_sample
            end_idx = start_idx + nodes_per_sample

            # Extracting the subgraph for this sample
            subgraph = processed_graph[start_idx:end_idx, :]

            # Loop over phi_atoms and psi_atoms to predict angles
            # Adjust indices for phi_atoms and psi_atoms based on batch
            phi_start_idx = b * angles_per_batch_phi
            phi_end_idx = phi_start_idx + angles_per_batch_phi
            psi_start_idx = b * angles_per_batch_psi
            psi_end_idx = psi_start_idx + angles_per_batch_psi

            for i, indices in enumerate(phi_atoms[phi_start_idx:phi_end_idx]):
                # Extract features for the group of four atoms for phi angle
                group_features = subgraph[indices, :].view(-1).clone()  # Flatten the features
                phi_tensor[b, i, 0] = self.angle_prediction_network_phi(group_features)

            for i, indices in enumerate(psi_atoms[psi_start_idx:psi_end_idx]):
                # Extract features for the group of four atoms for psi angle
                group_features = subgraph[indices, :].view(-1).clone()  # Flatten the features
                psi_tensor[b, i, 0] = self.angle_prediction_network_psi(group_features)

        phi_tensor = phi_tensor.view(-1, 1)  
        psi_tensor = psi_tensor.view(-1, 1)

        return phi_tensor, psi_tensor




###############################################################################################################################
    @staticmethod
    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta degrees.
        """
        axis = torch.nn.functional.normalize(axis, dim=0)
        a = torch.cos(torch.deg2rad(theta) / 2.0)
        b, c, d = -axis * torch.sin(torch.deg2rad(theta) / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return torch.stack([
            torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)]),
            torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)]),
            torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc])
        ])

    def apply_rotation(self, coords, axis_point1, axis_point2, current_angle, target_angle):
        """
        Apply a rotation around a defined axis to a set of coordinates to achieve the target angle.
        """
        # Calculate the angle difference that we need to rotate
        angle_difference = current_angle - target_angle
        # Get the rotation matrix for the desired rotation angle
        rot_matrix = self.rotation_matrix(axis_point2 - axis_point1, angle_difference)
        # Translate coords to origin, apply rotation, translate back
        translated_coords = coords - axis_point1
        rotated_coords = torch.matmul(rot_matrix, translated_coords.T).T
        return rotated_coords + axis_point1

    def update_positions(self, atom_coords, phi_atoms, psi_atoms, phi_angles, psi_angles):
        """
        Update the positions of atoms in a protein structure based on given phi and psi angles.
        """
        new_atom_coords = atom_coords.clone()
        for i in range(phi_atoms.shape[0]):
            i = 8-i
            current_phi_angle = calculate_all_torsion_angles_for_trajectories(new_atom_coords.unsqueeze(0).cpu(), phi_atoms, psi_atoms)['phi'][0][i]
            current_psi_angle = calculate_all_torsion_angles_for_trajectories(new_atom_coords.unsqueeze(0).cpu(), phi_atoms, psi_atoms)['psi'][0][i]
            # Apply phi rotation
            axis_phi = [new_atom_coords[phi_atoms[i, 1]], new_atom_coords[phi_atoms[i, 2]]]  # Axis defined by N-CA
            # axis_phi = [new_atom_coords[144], new_atom_coords[146]]
            for j in range(phi_atoms[i, 2].item(), new_atom_coords.shape[0]):  # Rotate everything after CA
                new_atom_coords[j] = self.apply_rotation(new_atom_coords[j], axis_phi[0], axis_phi[1], current_phi_angle, phi_angles[i])

            # SEE HERE if i.e., first phi_angle resemble the one that should be phi_angles[0]
            # print(calculate_all_torsion_angles_for_trajectories(new_atom_coords.unsqueeze(0).cpu(), phi_atoms, psi_atoms))
            # Apply psi rotation
            axis_psi = [new_atom_coords[psi_atoms[i, 1]], new_atom_coords[psi_atoms[i, 2]]]  # Axis defined by CA-C
            for j in range(psi_atoms[i, 2].item(), new_atom_coords.shape[0]):  # Rotate everything after C
                new_atom_coords[j] = self.apply_rotation(new_atom_coords[j], axis_psi[0], axis_psi[1], current_psi_angle, psi_angles[i])

        # print(calculate_all_torsion_angles_for_trajectories(new_atom_coords.unsqueeze(0).cpu(), phi_atoms, psi_atoms))
        # print(f"PHI:{phi_angles}")
        # print(f"PSI:{psi_angles}")
        return new_atom_coords

