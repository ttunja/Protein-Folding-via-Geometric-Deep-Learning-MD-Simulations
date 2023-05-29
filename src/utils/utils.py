import torch
from torch_cluster import radius_graph

EPSILON = 1e-8
ATOM_MASSES = [14.007, 1.008, 12.001, 16]

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
