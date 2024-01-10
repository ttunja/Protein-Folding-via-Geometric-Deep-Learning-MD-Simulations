import os
import pandas as pd

import time 
from pathlib import Path
# from preprocessing_utils import store_data
import copy
import numpy as np
import argparse

# import mdtraj
# from mdtraj import element

# TODO you have to change all datadir, so that you can specify it
# TODO automatically preprocess all trajectories in the directory

ATOM_TYPES = ['N', 'H', 'C', 'O']

def identify_torsion_atoms(table):
    """
    Identify torsion angles (phi, psi, omega) in a protein structure.

    Args:
    table (pd.DataFrame): A DataFrame containing atom data for the protein structure.
                          It must contain 'name', 'resSeq' columns.

    Returns:
    tuple: Three lists of atom indices for phi, psi, and omega angles, respectively.
    """

    # Validate input table
    if not {'name', 'resSeq'}.issubset(table.columns):
        raise ValueError("Input table must contain 'name' and 'resSeq' columns.")

    # Sort table by resSeq to handle non-continuous resSeq values
    table = table.sort_values(by='resSeq')

    phi_atoms, psi_atoms, omega_atoms = [], [], []

    # Create a dictionary for faster access
    grouped_table = table.groupby('resSeq')

    for resSeq, group in grouped_table:
        ca_atoms = group[group['name'] == 'CA'].index
        c_atoms = group[group['name'] == 'C'].index
        n_atoms = group[group['name'] == 'N'].index

        if not ca_atoms.empty and not c_atoms.empty and not n_atoms.empty:
            CA_curr = ca_atoms.item()
            C_curr = c_atoms.item()
            N_curr = n_atoms.item()

            # Handle phi angle
            if resSeq > table['resSeq'].min():
                prev_group = grouped_table.get_group(resSeq - 1)
                C_prev = prev_group[prev_group['name'] == 'C'].index
                if not C_prev.empty:
                    phi_atoms.append([C_prev.item(), N_curr, CA_curr, C_curr])

            # Handle psi angle
            if resSeq < table['resSeq'].max():
                next_group = grouped_table.get_group(resSeq + 1)
                N_next = next_group[next_group['name'] == 'N'].index
                if not N_next.empty:
                    psi_atoms.append([N_curr, CA_curr, C_curr, N_next.item()])

            # # Handle omega angle
            # if resSeq < table['resSeq'].max():
            #     CA_next = next_group[next_group['name'] == 'CA'].index
            #     if not CA_next.empty:
            #         omega_atoms.append([CA_curr, C_curr, N_next.item(), CA_next.item()])

    # return phi_atoms, psi_atoms, omega_atoms
    return phi_atoms, psi_atoms

    import numpy as np

def calculate_torsion_angle(atom1, atom2, atom3, atom4):
    """
    Calculate the torsion angle formed by four atoms.

    Args:
    atom1, atom2, atom3, atom4 (np.array): The coordinates of the four atoms.

    Returns:
    float: Torsion angle in degrees.
    """

    # Vectors between the atoms
    b1 = atom2 - atom1
    b2 = atom3 - atom2
    b3 = atom4 - atom3

    # Normal vectors to the planes formed by the first three and last three atoms
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    # Calculate the torsion angle
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.degrees(np.arctan2(y, x))


def calculate_all_torsion_angles_for_trajectories(atom_coords, phi_atoms, psi_atoms): #, omega_atoms):
    """
    Calculate all torsion angles for a given protein structure across multiple trajectories.

    Args:
    atom_coords (np.array): A 3D numpy array containing coordinates of atoms in the protein for multiple trajectories.
                            Shape should be (number of trajectories, number of atoms, 3).
    phi_atoms, psi_atoms, omega_atoms (list): Lists of tuples, each containing indices of atoms for phi, psi, and omega angles.

    Returns:
    dict: Dictionary containing lists of all phi, psi, and omega angles for each trajectory.
    """

    # Initialize dictionary to store angles for each trajectory
    # angles = {'phi': [], 'psi': [], 'omega': []}
    angles = {'phi': [], 'psi': []}

    # Iterate over each trajectory
    for trajectory in atom_coords:
        # traj_angles = {'phi': [], 'psi': [], 'omega': []}
        traj_angles = {'phi': [], 'psi': []}
        
        # for angle_list, angle_name in zip([phi_atoms, psi_atoms, omega_atoms], ['phi', 'psi', 'omega']):
        for angle_list, angle_name in zip([phi_atoms, psi_atoms], ['phi', 'psi']):
            for indices in angle_list:
                # Extract coordinates for the four atoms
                coords = [trajectory[i] for i in indices]
                angle = calculate_torsion_angle(*coords)
                traj_angles[angle_name].append(angle)

        for key in angles.keys():
            angles[key].append(traj_angles[key])

    return angles


def load_protein_data(data_dir, topology_dir):
    """
    atom_coords: float, (num_traj, num_atom, 3)
    atom_types: int, (num_atom,)
    lattices: float, (num_traj, 3)
    bond_indices: int, (num_bonds, 2)
    rgs: float, (num_traj,)
    """

    trajectory = mdtraj.load_dcd(data_dir, top=topology_dir)
    
    # trajectory = trajectory.center_coordinates(mass_weighted=False)
    # trajectory = trajectory.superpose(trajectory[0])
    # scale units to A
    atom_coords = trajectory.xyz * 10.

    # get topology of the graph
    table, bonds = trajectory.top.to_dataframe()

    phi_atoms, psi_atoms = identify_torsion_atoms(table)

    angles = calculate_all_torsion_angles_for_trajectories(atom_coords, phi_atoms, psi_atoms)

    table['radius'] = 0
    table['mass'] = 0
    for i in range(table.shape[0]):
        table['mass'].iloc[i] = element.get_by_symbol(table.element.iloc[i]).mass
        table['radius'].iloc[i] = element.get_by_symbol(table.element.iloc[i]).radius

    rgs = mdtraj.compute_rg(trajectory, masses=np.array(table.mass))*10


    atom_types = [ATOM_TYPES.index(atom) for atom in table.element]
    atom_types = np.array(atom_types)

    # shape (num_bonds, 2), denotes indices of atoms connected by bonds
    bond_indices = bonds[:, :2]
    mass = np.array(table.mass)

    assert atom_coords.shape[0] == rgs.shape[0]
    return [atom_coords, rgs, atom_types, bond_indices, mass, phi_atoms, psi_atoms, angles['phi'], angles['psi']]

def preprocess_traj(data_dir, topology_dir, data_save_dir):
    data_dir = Path(data_dir)
    topology_dir = Path(topology_dir)
    poly_file_dirs = [file for file in list(os.listdir(data_dir)) if file.endswith(".dcd")]
    print(f"Found {len(poly_file_dirs)} polymer trajectories.")
    print("Start processing...")
    for i in range(len(poly_file_dirs)):    
        data = load_protein_data(os.path.join(data_dir, poly_file_dirs[i]), topology_dir)
        os.makedirs(os.path.join(data_save_dir, f"traj_{i}"), exist_ok=True)
        # if not Path(str(os.path.join(data_save_dir, f"traj_{i}", 'bond.h5'))).exists():
        if True:
            store_data(['position'], [data[0]], os.path.join(data_save_dir, f"traj_{i}", 'position.h5'))
            store_data(['rgs'], [data[1]], os.path.join(data_save_dir, f"traj_{i}", 'rgs.h5'))
            store_data(['particle_type'], [data[2]], os.path.join(data_save_dir, f"traj_{i}", 'ptype.h5'))
            store_data(['bond_indices'], [data[3]], os.path.join(data_save_dir, f"traj_{i}", 'bond.h5'))
            store_data(['mass'], [data[4]], os.path.join(data_save_dir, f"traj_{i}", 'mass.h5'))
            store_data(['phi_atoms'], [data[5]], os.path.join(data_save_dir, f"traj_{i}", 'phi_atoms.h5'))
            store_data(['psi_atoms'], [data[6]], os.path.join(data_save_dir, f"traj_{i}", 'psi_atoms.h5'))
            store_data(['phi_angles'], [data[7]], os.path.join(data_save_dir, f"traj_{i}", 'phi_angles.h5'))
            store_data(['psi_angles'], [data[8]], os.path.join(data_save_dir, f"traj_{i}", 'psi_angles.h5'))
        else:
            print("Data already exists!")

    
if __name__=='__main__':

    # import ipdb; ipdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, required=True)
    parser.add_argument('--top_path', '-td', type=str, required=True)
    parser.add_argument('--save_dir', '-sd', type=str, required=True)

    args = parser.parse_args()
    data_dir = str(args.dir)
    topology_path = str(args.top_path)
    data_save_dir = str(args.save_dir)

    preprocess_traj(data_dir, topology_path, data_save_dir)
