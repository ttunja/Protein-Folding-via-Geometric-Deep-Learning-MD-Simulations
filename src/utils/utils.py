import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

EPSILON = 1e-8
ATOM_MASSES = [14.007, 1.008, 12.001, 16, 32.0655]

def get_position_noise(position_sequence, noise_std, use_rw=False):
    """
    Returns random-walk noise in the velocity applied to the position.
    For systems with PBC, takes in the unwrapped coordinates.
    """
    if use_rw:
        velocity_sequence = position_sequence[:, 1:] - position_sequence[:, :-1]
        num_velocities = velocity_sequence.shape[1]
        velocity_sequence_noise = (torch.randn_like(velocity_sequence) * 
                                (noise_std / num_velocities ** 0.5))
        velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
        position_sequence_noise = torch.cat([
            torch.zeros_like(velocity_sequence_noise[:, 0:1]),
            torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)
    else:
        position_sequence_noise = torch.randn_like(position_sequence) * noise_std

    return position_sequence_noise

def compute_connectivity(positions, n_node, radius, bonds=None, add_self_edges=True):
    batch = torch.arange(len(n_node)).to(n_node.device).repeat_interleave(n_node, dim=0)
    senders, receivers = radius_graph(positions, radius, batch, loop=add_self_edges)
    if bonds is not None:
        edge_indices = torch.cat([senders.unsqueeze(1), receivers.unsqueeze(1)], dim=1)
        all_edges = torch.cat([edge_indices, bonds], dim=0)
        all_edges, counts = torch.unique(all_edges, dim=0, return_counts=True)
        edge_types = (counts > 1).int()
        senders, receivers = all_edges[:, 0], all_edges[:, 1]
    else:
        edge_types = None
    # displacements normalized with radius.
    displacements = (positions[senders] - positions[receivers]) / radius
    distances = displacements.norm(dim=-1, keepdim=True)
    return senders, receivers, displacements, distances, edge_types

def compute_com(positions, weights, cluster):
    """
    compute center of mass according to the cluster assignments.
    Inputs:
        positions: N x T x dim
        weights: N x 1
        cluster: N x 1
    """
    weights = weights.view(-1, 1, 1)  
    cg_weights = scatter(weights.squeeze(), cluster, dim=0, reduce='sum').view(-1, 1, 1)
    cg_positions = scatter(weights*positions, cluster, dim=0, reduce='sum') / cg_weights
    cg_weights = cg_weights.view(-1, 1)
    return cg_positions, cg_weights

def coarse_graining(pos_seq, next_pos, ptype_embeddings, weights, 
                    use_keypoint_cg=False, cluster=None, keypoint=None):
    """
    coarse grain <pos_seq>, <next_pos>, <ptype_embeddings>, and <weights> using the 
    fine -> coarse mapping given by <cluster>.
    for systems with PBC only use the **unwrapped** positions as inputs.
    """
    pos_seq = torch.cat([pos_seq, next_pos[:, None, :]], dim=1)
    if use_keypoint_cg:
        cg_pos_seq = pos_seq[keypoint]
        weights = weights.view(-1, 1, 1)  
        cg_weights = scatter(weights.squeeze(), cluster, dim=0, reduce='sum').view(-1, 1)
    else:
        cg_pos_seq, cg_weights = compute_com(pos_seq, weights, cluster)
    cg_pos_seq, cg_next_pos = cg_pos_seq[:, :-1], cg_pos_seq[:, -1]
    cg_ptype_embeddings = scatter(ptype_embeddings, cluster, dim=0, reduce='mean')
    return cg_pos_seq, cg_next_pos, cg_ptype_embeddings, cg_weights


def compute_weighted_rgs(positions, weights, n_node):
    """
    Inputs:
        positions: N x dim
        weights: N x 1
        n_node: B x 1
    Outputs:
        rgs: B x 1
    """
    weights = weights.view(-1, 1)
    positions_per_graph_list = torch.split(positions, list(n_node), dim=0)
    weights_per_graph_list = torch.split(weights, list(n_node), dim=0)
    all_rgs = []
    for positions, weights in zip(positions_per_graph_list, weights_per_graph_list):
        com = (positions * weights).sum(0) / weights.sum()
        rgs = (weights.squeeze()*((positions - com).pow(2)).sum(1)).sum() / weights.sum()
        rgs = rgs.sqrt()
        all_rgs.append(rgs.view(-1))
    return torch.cat(all_rgs).unsqueeze(1)
