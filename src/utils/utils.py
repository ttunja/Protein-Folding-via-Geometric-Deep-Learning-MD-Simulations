import numpy as np
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from scipy.stats import entropy

EPSILON = 1e-8
PSEUDO_MASSES = [1.] * 10
# ATOM_MASSES = [14.007, 1.008, 12.001, 16, 32.0655]

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

def compute_connectivity(positions, n_node, radius, bonds=None, add_self_edges=True, senders=None, receivers=None):
    batch = torch.arange(len(n_node)).to(n_node.device).repeat_interleave(n_node, dim=0)
    if senders is None or receivers is None:
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


def compute_autocorrelation(accelerations, max_lag=None):
    """
    Computes the autocorrelation for 3D vector accelerations across time.

    Args:
        accelerations (torch.Tensor): Tensor of shape [B, T, 3] where B is the number of beads,
                                      T is the number of time frames, and 3 are the x, y, z components of acceleration.
        max_lag (int): Maximum lag to compute autocorrelation for. Default is None, which sets max_lag to T-1.

    Returns:
        autocorr (torch.Tensor): Autocorrelation values for each lag from 0 to max_lag.
    """
    B, T, _ = accelerations.shape
    if max_lag is None:
        max_lag = T - 1

    dot_product_at_0 = (accelerations * accelerations).sum(dim=-1)  # [B, T]
    avg_dot_product_at_0 = dot_product_at_0.mean(dim=1)  # Average over time frames for each bead
    normalization_factor = avg_dot_product_at_0.mean()  # Overall average across all beads

    autocorr = []
    for tau in range(max_lag + 1):
        # Dot product of accelerations at time t and t + tau
        dot_product = (accelerations[:, :-tau] * accelerations[:, tau:]).sum(dim=-1) if tau > 0 else (accelerations * accelerations).sum(dim=-1)
        
        # Average over beads and time frames
        avg_dot_product = dot_product.mean(dim=1)
        autocorr_value = avg_dot_product.mean() / normalization_factor  # Normalize by A(t_0)^2

        autocorr.append(autocorr_value)

    autocorr = torch.tensor(autocorr)  # Convert list to tensor
    return autocorr


def compute_distance_matrix(x):
    """
    x: Tensor of shape (N, 3) where N is the number of atoms.
    This function returns a matrix of shape (N, N) where the entry at (i, j) is
    the Euclidean distance between atoms i and j.
    """
    # Expand x along the second dimension
    x = x.unsqueeze(1)

    # Calculate the pairwise distance in one line by taking advantage of broadcasting
    dist = torch.sqrt(torch.sum((x - x.transpose(0, 1)) ** 2, -1)+1e-8)

    return dist

def KL_divergence(true_density, pred_density):
    # Normalize the densities for KL divergence (important step)
    true_density_flat = true_density.ravel() / np.sum(true_density)
    pred_density_flat = pred_density.ravel() / np.sum(pred_density)

    # Add a small epsilon to both distributions to avoid log(0)
    true_density_flat += 1e-10
    pred_density_flat += 1e-10

    # Compute KL divergence between the true and predicted KDEs
    kl_div = entropy(true_density_flat, pred_density_flat)
    return kl_div


def compute_dihedral_angle(p1, p2, p3, p4):
    """
    Compute the dihedral angle between four points p1, p2, p3, p4 (3D coordinates).
    Returns the angle in the range of -180 to 180 degrees.
    """
    # Create vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Normal vectors to the planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    # Unit vectors for the normal vectors
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    
    # Normalize b2 for the direction of the angle
    b2 /= np.linalg.norm(b2)
    
    # Calculate an auxiliary vector perpendicular to the plane of b2
    m1 = np.cross(n1, b2)
    
    # Angle in radians
    angle = np.arctan2(np.dot(m1, n2), np.dot(n1, n2))
    
    # Convert to degrees and return
    return np.degrees(angle)

def compute_ramachandran_angles(bead_positions):
    """
    Compute pseudo phi and psi angles for a set of bead positions over multiple frames.
    bead_positions: shape [N_beads, N_frames, 3]
    Returns:
    - phi_angles: Pseudo phi angles for each frame
    - psi_angles: Pseudo psi angles for each frame
    """
    N_beads, N_frames, _ = bead_positions.shape
    
    # Initialize lists to store the angles
    phi_angles = []
    psi_angles = []
    
    # Loop over each frame to calculate angles
    for frame in range(N_frames):
        # i = 3
        for i in range(N_beads - 4):  # We need 4 consecutive beads for dihedral angles
            p1 = bead_positions[i, frame, :]
            p2 = bead_positions[i + 1, frame, :]
            p3 = bead_positions[i + 2, frame, :]
            p4 = bead_positions[i + 3, frame, :]
            p5 = bead_positions[i + 4, frame, :]
            
            # Calculate phi (dihedral between beads i, i+1, i+2, i+3)
            phi = compute_dihedral_angle(p1, p2, p3, p4)
            phi_angles.append(phi)
            
            # For simplicity, we can define psi similarly, or offset by one bead
            # For now, use the same approach for pseudo-psi
            psi = compute_dihedral_angle(p2, p3, p4, p5)  # Adjust bead selection as needed
            psi_angles.append(psi)
    
    return np.array(phi_angles), np.array(psi_angles)
    