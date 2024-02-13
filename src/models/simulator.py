import models.graphs as graphs
from models.TINGnn import EncodeProcessDecode, GraphPooling, build_mlp
import math
import hydra
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from utils import utils
from utils.utils import coarse_graining
from tqdm import tqdm

STD_EPSILON = 1e-8


class LearnedSimulator(nn.Module):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(
        self,
        connectivity_radius,
        graph_network_kwargs,
        embedding_network_kwargs,
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
        self.use_coarse_graining = True#(self.hparams.cg_level > 1) or ('use_keypoint_for_cg' in self.hparams and self.hparams.use_keypoint_for_cg)
        self.embedding_network_kwargs = embedding_network_kwargs
        self.particle_type_embedding_size = particle_type_embedding_size
        self.edge_type_embedding_size = edge_type_embedding_size
        self.graph_network_kwargs = graph_network_kwargs

        self.embedding_gn = EncodeProcessDecode(
                                            node_dim=self.particle_type_embedding_size, #+ 1,
                                            edge_dim=self.edge_type_embedding_size,  
                                            out_dim=128, 
                                            # output_size=32,
                                            **self.embedding_network_kwargs)

        self.type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        
        # types of fine level bonds. set <num_btypes==0> to disable bond type information.
        if num_bond_types > 0:
            self.bond_type_embedding = nn.Embedding(num_bond_types, particle_type_embedding_size)
            nn.init.xavier_uniform_(self.bond_type_embedding.weight)
        
        self.edge_type_embedding = nn.Embedding(2, edge_type_embedding_size)  # 2 types: real bonds vs radius edge.   
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)

        self.graph_pool = GraphPooling(reducers=['mean'])
        self.property_head = build_mlp(in_dim=128, 
                                       units=256, 
                                       layers=2, 
                                       out_dim=1)

        self.sigmas = torch.linspace(math.log(0.1), 
                                    math.log(0.01), 
                                    10).exp().to('cuda')
        self.sigmas.requires_grad = False
        
    def forward(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError

    def simulate(self, batch, rollout_length, 
               deter=False, save_positions=False, save_frequency=1,
               disable_bar=False):
        """
        simulate CGMD from a data batch of length <rollout_length>.
        deter:          if <True>, predict deterministically.
        save_positions: if <True>, save the rollout bead positions.
        save_frequency: frameskip when recording the trajectory.
        disable_bar:    if <True>, disable the progress bar.
        """
        output = {}
        position_seq, next_pos = batch['position'], batch['target']
        position_seq = position_seq[:, :30]
        if self.particle_masses.device != position_seq.device:
            self.particle_masses = self.particle_masses.to(position_seq.device)
        
        n_node, ptypes = batch['n_particle'], batch['particle_types']
        bonds = batch['bonds']
        bond_types = batch.get('bond_types')
        weights = self.particle_masses[ptypes]
        ptype_embeddings = self._encoder_preprocessor(ptypes, n_node, bonds, bond_types, weights)
        use_keypoint_cg = False
        if self.use_coarse_graining:
            (position_seq, next_pos, ptype_embeddings, weights) = coarse_graining(
                position_seq, next_pos, ptype_embeddings, weights, use_keypoint_cg,
                batch.get('cluster'), batch.get('keypoint'))

            n_node = batch.get('n_keypoint')
            bonds = batch.get('cg_bonds')
        output['cg_u_pos'] = position_seq.detach().clone().cpu()
        output['cg_weights'] = weights
        
        all_positions = []
        all_props = []
        for t in tqdm(range(rollout_length), disable=disable_bar):
            next_pos_pred, prop_pred = self.predict(
                position_seq, ptype_embeddings, n_node, bonds, bond_types, weights)
            position_seq = torch.cat([position_seq[:, 1:], next_pos_pred.unsqueeze(1)], dim=1)
            if t % save_frequency == 0:
                all_positions.append(next_pos_pred.detach().cpu().unsqueeze(1))
            if prop_pred is not None:
                all_props.append(prop_pred.detach().cpu())

        if save_positions:
            output['rollout_u_pos'] = torch.cat(all_positions, dim=1)
        
        if prop_pred is not None:
            output['rollout_prop'] = torch.cat(all_props, dim=1)
        
        return output

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
            
            # if weights is not None:
            #     particle_type_embeddings = torch.cat([particle_type_embeddings, weights.view(-1, 1)], dim=-1)
                
            graph_tuple = graphs.GraphsTuple(
                nodes=particle_type_embeddings,
                edges=edge_embeddings,
                edge_type=None,
                coords=None,
                globals=None,
                n_node=n_node,
                senders=senders,
                receivers=receivers)

            return self.embedding_gn(graph_tuple)
    
    def _dynamics_preprocessor(self, position_seq, particle_type_embeddings, n_node,
                               bonds=None, bond_types=None, weights=None):
        
        velocity_sequence = self.time_diff(position_seq)  # Finite-difference.
        flat_vel_seq = torch.reshape(velocity_sequence, 
                                    [velocity_sequence.shape[0], 
                                     math.prod(velocity_sequence.shape[1:])])
        # if weights is not None:
        #     node_features = torch.cat([flat_vel_seq, particle_type_embeddings, weights.view(-1, 1)], dim=-1)
        # else:
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

    def time_diff(self, input_sequence):
        return input_sequence[:, 1:] - input_sequence[:, :-1]

    def predict_prop(self, latent_graph, pos, weights, n_node, prop='rgs'):
        """
        predict the the specified property if applicable.
        For certain properties, we can compute from the CG state and fit the
        residual using a neural network on top of the latent graph.
        """
        pooled_graph = self.graph_pool(latent_graph.nodes.detach(), 
                                    latent_graph.n_node.detach())
        pred = self.property_head(pooled_graph)
        if prop == 'rgs':
            pred = pred + utils.compute_weighted_rgs(pos.detach(), weights.detach(), n_node)
        return pred

