import torch
from torch_scatter import scatter
from torch_cluster import radius_graph

EPSILON = 1e-8

PSEUDO_MASSES = [1.] * 100

ATOM_MASSES = [
    0.0, 1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999,
    18.998403163, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983, 40.078, 44.955908,
    47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62,
    88.90584, 91.224, 92.90637, 95.95, 97.90721, 101.07, 102.9055, 106.42,
    107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293,
    132.90545196, 137.327, 138.90547, 140.116, 140.90766, 144.242, 144.91276,
    150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422,
    173.045, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217,
    195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804, 209.0, 210.0,
    222.0, 223.0, 226.0, 227.0, 232.0377, 231.03588, 238.02891, 237.0, 244.0,
    243.0, 247.0, 247.0, 251.0, 252.0]

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
