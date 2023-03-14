import os
import pandas as pd
import mdtraj
import time 
from pathlib import Path
from utils_tin import store_data, topology_to_json, load_data
import copy
import numpy as np
import argparse
from mdtraj import element

# TODO you have to change all datadir, so that you can specify it
# TODO automatically preprocess all trajectories in the directory

ATOM_TYPES = ['N', 'H', 'C', 'O']

def load_chignolin_rg(data_dir):
    """
    atom_coords: float, (num_traj, num_atom, 3)
    atom_types: int, (num_atom,)
    lattices: float, (num_traj, 3)
    bond_indices: int, (num_bonds, 2)
    rgs: float, (num_traj,)
    """
    # path = "D:\PROTEIN-FOLDING-DATA\Chignolin\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein"
    # trajectory = trajectory = mdtraj.load_dcd(path+"\CLN025-0-protein-002.dcd", 
    #                                           top=path+"\Topology.pdb")
    trajectory = trajectory = mdtraj.load_dcd(data_dir, 
                                              top="D:\PROTEIN-FOLDING-DATA\Chignolin\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein\Topology.pdb")
    
    trajectory = trajectory.center_coordinates(mass_weighted=False)
    trajectory = trajectory.superpose(trajectory[0])
    # scale units to A
    atom_coords = trajectory.xyz * 10.
    lattices = trajectory.unitcell_lengths * 10.

    # get topology of the graph
    table, bonds = trajectory.top.to_dataframe()

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

    assert atom_coords.shape[0] == lattices.shape[0] == rgs.shape[0]
    return [atom_coords, lattices, rgs, atom_types, bond_indices]

def preprocess_traj(data_dir, data_save_dir):
    data_dir = Path(data_dir)
    poly_file_dirs = [file for file in list(os.listdir(data_dir)) if file.endswith(".dcd")]
    print(f"Found {len(poly_file_dirs)} polymer trajectories.")
    print("Start processing...")
    for i in range(len(poly_file_dirs)):    
        data = load_chignolin_rg(os.path.join(data_dir, poly_file_dirs[i]))
        # datadir = "D:\PROTEIN-FOLDING-DATA\Chignolin\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein" # CHANGE
        os.makedirs(os.path.join(data_save_dir, f"traj_{i}"), exist_ok=True)
        if not Path(str(os.path.join(data_save_dir, f"traj_{i}", 'bond.h5'))).exists():
            store_data(['position'], [data[0]], os.path.join(data_save_dir, f"traj_{i}", 'position.h5'))
            store_data(['lattice'], [data[1]], os.path.join(data_save_dir, f"traj_{i}", 'lattice.h5'))
            store_data(['rgs'], [data[2]], os.path.join(data_save_dir, f"traj_{i}", 'rgs.h5'))
            store_data(['particle_type'], [data[3]], os.path.join(data_save_dir, f"traj_{i}", 'ptype.h5'))
            store_data(['bond_indices'], [data[4]], os.path.join(data_save_dir, f"traj_{i}", 'bond.h5'))
        else:
            print("Data already exists!")
    # process_one_file(poly_file_dirs[0])

def read_h5_as_traj(datadir):
    import mdtraj
    from mdtraj.formats import DCDTrajectoryFile
    from mdtraj import element
    import nglview as nv
    import os
    os.chdir("D:\PROTEIN-FOLDING-DATA\Chignolin\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein")
    trajectory = mdtraj.load_dcd("CLN025-0-protein-000.dcd", top="Topology.pdb")
    topology_data = topology_to_json(trajectory.topology)
    data = load_chignolin_rg(datadir)
    traj = mdtraj.load_hdf5("C:/Users/ttunj/Downloads/new_try_traj.h5")
    # traj = mdtraj.load_hdf5("test_try_set/coordinates.h5")#, top="D:/PROTEIN-FOLDING-DATA/Chignolin/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/Topology.pdb")
    # new = nv.show_mdtraj(traj)
    # data_old = load_data(['coordinates', 'topology'], 'test_set\coordinates.h5')
    # data = load_data(['coordinates', 'topology'], 'test_try_set\coordinates.h5')
    datadir = "D:\PROTEIN-FOLDING-DATA\Chignolin\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein" # CHANGE
    os.makedirs(os.path.join(datadir, "real_try_set"), exist_ok=True)
    if not Path(str(os.path.join(datadir, "real_try_set", 'bond.h5'))).exists():
        store_data(['coordinates'], [data[0]], os.path.join(datadir, "real_try_set", 'coordinates.h5'))
    return trajectory

#"C:\Users\ttunj\OneDrive\Desktop\mlcgmd_models\chain_gns\predicted_traj\coordinates.h5"
# mdconvert "C:\Users\ttunj\OneDrive\Desktop\mlcgmd_models\chain_gns\new_predicted_traj\coordinates.h5" -o new_traj_longer.h5 -t "D:\PROTEIN-FOLDING-DATA\Chignolin\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein\Topology.pdb" 

    
if __name__=='__main__':

    import ipdb; ipdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, required=True)
    parser.add_argument('--save_dir', '-sd', type=str, required=True)

    args = parser.parse_args()
    data_dir = str(args.dir)
    data_save_dir = str(args.save_dir)

    # read_h5_as_traj(datadir)
    preprocess_traj(data_dir, data_save_dir)